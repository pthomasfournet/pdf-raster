# pdf-raster — Architecture

End-to-end description of the codebase: crate topology, data flow, key
abstractions, and platform strategy.

---

## 1. Crate topology

```
color
  │   pixel types, colour math, convert helpers
  │
  ├── raster
  │     software rasterizer — paths, fills, compositing, SIMD
  │     │
  │     ├── font
  │     │     FreeType glyph cache + outline → Path bridge
  │     │
  │     ├── encode
  │     │     PPM / PGM / PBM / PNG output
  │     │
  │     └── pdf_interp
  │           PDF content stream interpreter + operator dispatcher
  │           │
  │           ├── gpu  (optional, feature-gated)
  │           │     CUDA kernels, nvJPEG/nvJPEG2k decoders, NPP rotate
  │           │
  │           └── pdf_raster
  │                 public OCR library — raster_pdf(), render_channel()
  │                 │
  │                 └── cli
  │                       pdf-raster binary (pdftoppm replacement)
  │
  ├── pdf_bridge  (unused in render path; poppler reference baseline only)
  │
  └── bench  (vello_cpu throughput comparison; not linked by cli)
```

`color` and `raster` are the foundation. Nothing above them has knowledge of PDF.
`pdf_interp` is the only crate that parses PDF; everything below it is PDF-agnostic.

---

## 2. Data flow — one page, start to finish

```
PDF bytes on disk
    │
    ▼ lopdf::Document
pdf_interp::open()
    │
    ▼ Vec<Operator>
pdf_interp::parse_page()             content tokenizer → operator decode
    │
    ▼ Bitmap<Rgb8>  (blank, DPI-sized)
pdf_interp::PageRenderer::new()      allocates target bitmap
    │
    ▼
render_operators(ops, resources)     dispatch loop:
    ├── path ops  ──────────────────▶ raster::fill / stroke / eo_fill
    ├── text ops  ──────────────────▶ font::FontFace → GlyphBitmap
    │                                 raster::fill_glyph
    ├── image ops ──────────────────▶ resources::resolve_image
    │   DCT      ───────────────────▶ gpu::NvJpegDecoder  (or zune-jpeg)
    │   JPX      ───────────────────▶ gpu::NvJpeg2kDecoder (or jpeg2k)
    │   CCITT/JBIG2 ────────────────▶ fax / hayro-jbig2
    │   Flate    ───────────────────▶ miniz_oxide (via lopdf)
    │   CMYK     ───────────────────▶ gpu::icc_cmyk_to_rgb (or AVX-512/scalar)
    │                                 raster::draw_image
    ├── shading ops ────────────────▶ raster::shaded_fill / gouraud_triangle_fill
    ├── tiling ops  ────────────────▶ raster::TiledPattern
    └── transparency groups ────────▶ raster begin_group / paint_group
    │
    ▼ Bitmap<Rgb8>
pdf_raster::rgb_to_gray()            BT.709 luma
    │
    ▼ Vec<u8>  (width × height, 8-bit grayscale)
RenderedPage { pixels, width, height, dpi, effective_dpi, diagnostics }
    │
    ▼
deskew (optional)                    projection-profile sweep ±7°
    │   gpu::npp_rotate  (or cpu bilinear fallback)
    │
    ▼
caller / Tesseract OCR
```

No file I/O after the initial `open()`. No subprocess at any stage.

---

## 3. Crates in detail

### 3.1 `color`

Single source of truth for pixel types and colour math. No SIMD, no I/O.

**Types**
- `Pixel` trait — `Copy + Pod + Zeroable + Send + Sync`
- Concrete: `Rgb8`, `Rgba8`, `Gray8`, `Cmyk8`, `DeviceN8`, `AnyColor`
- `PixelMode` enum — `Mono1 | Mono8 | Rgb8 | Bgr8 | Xbgr8 | Cmyk8 | DeviceN8`

**Primitives** (all in `convert`)
- `div255(u32) → u8` — fast ≈255 division, used throughout compositing
- `lerp_u8`, `over_u8` — bilinear and Porter-Duff source-over per channel
- `cmyk_to_rgb`, `cmyk_to_rgb_reflectance` — subtractive complement
- `byte_to_col` / `col_to_byte` — 16.16 fixed-point GfxColorComp bridge
- `TransferLut` — `[u8; 256]` newtype for transfer functions

### 3.2 `raster`

~9 000 lines. The software rendering engine. PDF-agnostic — takes paths,
bitmaps, and compositing parameters; knows nothing about operators or resources.

**`Bitmap<P: Pixel>`**
The primary output container. Width, height, stride, `Vec<u8>` data, optional
separate alpha plane. Generic over `P` — the compiler generates one code path
per pixel format with no runtime mode dispatch in hot loops.

**Path pipeline**
```
PathBuilder  →  Path  →  flatten_path()  →  XPath (edge table)
                          make_stroke_path()      │
                                                  ▼
                                           XPathScanner  →  ScanIterator
                                                            (x0, x1, y) spans
```
Bézier flattening via De Casteljau. Stroke expansion produces a fill path from
the outline — strokes and fills share a single rasterizer.

**Fill rasterizer**
Active edge table scan, 4× supersampled AA (`AA_SIZE = 4`). Coverage per output
pixel = popcount of 4 nibbles across 4 AA sub-rows. Hot path:
`aa_coverage_span()` → SIMD popcount tier (see §5).

Three compositing pipeline variants, selected at paint time by `PipeState::is_simple_path()`:

| Variant | When |
|---|---|
| `simple` | `a_input=255`, Normal blend, no soft mask, no group |
| `aa` | Shape byte present, Normal blend, no soft mask, no group correction |
| `general` | Everything else — soft mask, blend modes, transparency groups |

**Transparency groups**
`begin_group` / `paint_group` — allocate an isolated sub-bitmap, composite back
into the parent using the group's blend mode and opacity. Knockout groups supported.

**Pattern types**
- `TiledPattern` — implements `Pattern` trait; fills spans via repeated sub-bitmap blits
- Shading patterns — axial, radial, function-based, Gouraud triangle mesh

**SIMD module** (`simd/`)
Per-arch dispatch helpers; each arch gets its own function, not a cfg/return chain:
```
blend_solid_rgb8 / blend_solid_gray8  (movdir64b → AVX2 → NEON → scalar)
composite_aa_rgb8                     (AVX-512 → scalar)
popcnt_aa_row                         (AVX-512 VPOPCNTDQ → AVX2 → popcnt → SVE2 → NEON → scalar)
aa_coverage_span                      (AVX-512 BITALG → AVX2 → SVE2 → NEON → scalar)
unpack_mono_row                       (SSE2 → NEON → scalar)
```
SVE2 tier requires `nightly-sve2` Cargo feature (nightly Rust + `stdarch_aarch64_sve`).

**Parallel fill** (feature: `rayon`)
`fill_parallel` / `eo_fill_parallel` split the bitmap into horizontal bands above
`PARALLEL_FILL_MIN_HEIGHT`. Each band is an independent `BitmapBand` (`&mut` slice),
no synchronisation needed.

### 3.3 `font`

FreeType wrapper with two-level caching.

- `FontEngine` — owns `FT_Library`, assign face IDs. Protected by `Arc<Mutex<_>>`
  (FreeType is not thread-safe; lock is coarse but held only during face init, not render)
- `FontFace` — scaled face; `make_glyph(code) → GlyphBitmap`
- `GlyphCache` — `DashMap<GlyphKey, GlyphBitmap>` (lock-free, shared globally)
- `GlyphKey` — `{ face_id, code, size_px, base_idx, aa }`
- `decompose_outline(ft_outline) → raster::Path` — bridge from FreeType contours
  to the raster path format; glyphs rendered as path fills when AA is on, direct
  bitmap blit when AA is off

### 3.4 `encode`

Thin I/O layer. Takes `Bitmap<P>`, writes file format bytes to a `Write` sink.

- `write_ppm` — Netpbm P6, converts CMYK/DeviceN to RGB via subtractive complement
- `write_pgm` — Netpbm P5 grayscale
- `write_pbm` — Netpbm P4 1-bit
- `write_png` — `png` crate; supports Rgb8, Gray8, Rgba8

No pixel processing beyond colour-mode conversion at the boundary.

### 3.5 `pdf_interp`

PDF parsing and operator dispatch. The only crate that touches lopdf.

**Sub-modules**

```
content/
  tokenizer.rs     — byte-level PDF token scanner
  operands.rs      — operand stack + type coercion
  operator.rs      — 150+ typed operator variants
  mod.rs           — parse(bytes) → Vec<Operator>

renderer/
  mod.rs           — PageRenderer; render_operators() dispatch loop
  gstate.rs        — CTM, graphics state, StateStack
  color.rs         — colour space resolution + transfer functions
  font_cache.rs    — per-session face loading + GlyphCache integration
  text.rs          — text matrix, font state
  page/            — page-level entry points, geometry normalisation
  gpu_ops.rs       — GPU dispatch thresholds + optional gpu:: calls

resources/
  mod.rs           — PageResources; lazy resolve_*
  font.rs          — font descriptor parsing
  image/           — image descriptor, filter detection, decode dispatch
  icc.rs           — ICC profile loading + moxcms CLUT extraction
  shading/         — axial/radial/mesh shading descriptors
  cmap.rs          — ToUnicode / CMap parsing
  dict_ext.rs      — lopdf dict helper trait
  tiling.rs        — Type 1 tiling pattern parsing
```

**PageResources** — holds an immutable `&Document` reference. All resource
resolution is stateless and on-demand; no per-page pre-computation. This
makes unit testing straightforward: inject a `Document` and page `ObjectId`.

**GPU integration points** (feature-gated, all in `renderer/gpu_ops.rs`)
- DCTDecode: call `gpu::NvJpegDecoder::decode()` if pixels ≥ `GPU_JPEG_THRESHOLD_PX`
- JPXDecode: call `gpu::NvJpeg2kDecoder::decode()` similarly
- CMYK→RGB: call `gpu::icc_cmyk_to_rgb()` if pixels ≥ `GPU_ICC_CLUT_THRESHOLD`
- AA fill: `gpu::aa_fill()` / `gpu::tile_fill()` if span ≥ `GPU_AA_FILL_THRESHOLD`

GPU decoders live in thread-local storage on rayon worker threads (one per thread).
They are torn down eagerly via `pdf_raster::release_gpu_decoders()` broadcast before
the rayon pool drops — this avoids the CUDA driver teardown race at process exit.

**Error handling**
- Malformed operators — silently discarded (lenient parsing; real PDFs have junk)
- Missing resources — `InterpError::MissingResource`; page fails, next page continues
- JavaScript — rejected at open time (`InterpError::JavaScript`)

### 3.6 `gpu`

All CUDA code. Not linked unless at least one `gpu-*` feature is active.

**`GpuCtx`** — one per process, holds the CUDA context and all compiled PTX kernels
as `cudarc` modules. Shared via `Arc<GpuCtx>`.

**Kernel inventory**

| Kernel | File | Operation | Threshold |
|---|---|---|---|
| `composite_rgba8` | composite_rgba8.cu | Porter-Duff source-over | 500K px |
| `apply_soft_mask` | (same) | Per-pixel alpha multiply | 500K px |
| `aa_fill` | aa_fill.cu | 64-sample jittered AA coverage (warp ballot) | 256 px |
| `tile_fill` | tile_fill.cu | Analytical 16×16 tile fill | 256 px |
| `icc_clut` | icc_clut.cu | CMYK→RGB via 4D quadrilinear CLUT | 500K px |
| `icc_clut` | icc_clut.cu | CMYK→RGB via matrix (always CPU) | — |

**Optional decoders** (feature-gated)
- `nvjpeg` — `NvJpegDecoder`, TLS one-per-thread, primary `GPU_HYBRID`, fallback `DEFAULT`
- `nvjpeg2k` — `NvJpeg2kDecoder`, same TLS pattern; C++ exception shim in `shim/nvjpeg2k_shim.cpp`
- `gpu-deskew` — `npp_rotate()` via `nppiRotate_8u_C1R_Ctx`

**CPU fallbacks** — every GPU function has a pure-Rust CPU counterpart. The
dispatch logic is in the same function; the threshold is the only branch.

**CMYK CPU path** — three tiers: AVX-512 (`avx512f+avx512bw`, runtime detection),
AVX2 (`_mm256_mullo_epi16`, runtime detection), scalar. All three are compiled in
on x86-64; the correct tier is selected at runtime. ARM uses `vmull_u8` + `vshrn_n_u16`
(NEON, always-on on aarch64).

### 3.7 `pdf_raster`

Public library crate. The stable API surface.

```rust
RasterOptions { dpi, first_page, last_page, deskew }

raster_pdf(path, opts) → impl Iterator<Item = (u32, Result<RenderedPage, RasterError>)>
render_channel(path, opts, capacity) → Receiver<(u32, Result<RenderedPage, RasterError>)>
release_gpu_decoders()   // call via pool.broadcast() before pool drops
```

`RenderedPage` carries: `pixels: Vec<u8>` (8-bit gray, width×height), `width`,
`height`, `dpi`, `effective_dpi` (= dpi × UserUnit), and `PageDiagnostics`.

`PageDiagnostics::suggested_dpi(min, max)` — hint for re-rendering at native image
resolution. Returns `None` for vector/text-only pages.

`render_channel` spawns a Rayon background task and feeds pages into a bounded
`sync_channel`. Typical use: `capacity=4`, Tesseract consumes from the other end
while the next page renders.

### 3.8 `cli`

Binary entry point. Thin glue over `pdf_raster` + `encode`.

```
clap args → RasterOptions → rayon ThreadPool
  → pool.install(|| render_pages())
  → pool.broadcast(release_gpu_decoders)    // teardown before pool drops
  → encode::write_ppm / write_png per page
```

All rendering is inside `pool.install()`. The broadcast teardown is the last thing
that happens before the pool is dropped, ensuring GPU decoders are released while
the CUDA driver is fully live.

---

## 4. Feature flag graph

```
cli feature        → pdf_raster feature → pdf_interp feature → gpu feature
───────────────────────────────────────────────────────────────────────────
nvjpeg             → nvjpeg             → nvjpeg             → nvjpeg
nvjpeg2k           → nvjpeg2k           → nvjpeg2k           → nvjpeg2k
gpu-aa             → gpu-aa             → gpu-aa             → (enables gpu module)
gpu-icc            → gpu-icc            → gpu-icc            → (enables cmyk module)
gpu-deskew         → gpu-deskew         →                    → gpu-deskew
gpu-validation     →                    →                    → gpu-validation

raster features (orthogonal, no cross-crate effect):
  simd-avx2        default on; enables blend/fill AVX2 paths
  simd-avx512      implies simd-avx2; adds VPOPCNTDQ AA counter
  nightly-sve2     aarch64 only; SVE2 popcount tier (requires nightly Rust)
  rayon            enables fill_parallel / eo_fill_parallel
```

`dep:gpu` in `pdf_interp` is optional — the `gpu` crate is only compiled when
at least one `gpu-*` feature is active. CPU-only builds require no CUDA toolkit.

---

## 5. SIMD and platform strategy

### Dispatch pattern

All public SIMD functions delegate to a private per-arch `dispatch_*` helper.
Each arch gets its own clean function; no cfg/return fallthrough chains.

```rust
pub fn popcnt_aa_row(row: &[u8]) -> u32 { dispatch_popcnt(row) }

#[cfg(target_arch = "x86_64")]
fn dispatch_popcnt(row: &[u8]) -> u32 { /* avx512 → popcnt → scalar */ }

#[cfg(target_arch = "aarch64")]
fn dispatch_popcnt(row: &[u8]) -> u32 {
    // SVE2 tier above NEON when `nightly-sve2` feature is active
    #[cfg(feature = "nightly-sve2")]
    if is_aarch64_feature_detected!("sve2") { return unsafe { popcnt_aa_row_sve2(row) }; }
    unsafe { popcnt_aa_row_neon(row) }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn dispatch_popcnt(row: &[u8]) -> u32 { popcnt_aa_row_scalar(row) }
```

### Coverage by operation and platform

| Operation | File | Ryzen (AVX-512) | Intel consumer (AVX2) | ARM NEON | ARM SVE2 |
|---|---|---|---|---|---|
| AA row popcount | `raster/simd/popcnt.rs` | VPOPCNTDQ | AVX2 VPSHUFB | `vcntq_u8` | `svcnt_u8_z` † |
| AA coverage span | `raster/simd/popcnt.rs` | BITALG | AVX2 VPSHUFB | `vcntq_u8` + `vst2q_u8` | `svcnt_u8_z` † |
| Solid fill (RGB) | `raster/simd/blend.rs` | `movdir64b` → AVX2 | AVX2 | `vst3q_u8` | (NEON sufficient) |
| Solid fill (Gray) | `raster/simd/blend.rs` | AVX2 | AVX2 | `vst1q_u8` | (NEON sufficient) |
| AA composite | `raster/simd/composite.rs` | AVX-512 | scalar | scalar | scalar |
| Glyph unpack | `raster/simd/glyph_unpack.rs` | SSE2 | SSE2 | `vtstq_u8` | (NEON sufficient) |
| CMYK→RGB | `gpu/src/cmyk.rs` | AVX-512 | AVX2 | `vmull_u8` + `vshrn_n_u16` | (NEON sufficient) |

† SVE2 tier requires `nightly-sve2` Cargo feature.

### Key platform facts

**x86-64 (Intel consumer)**
- AVX2 is the ceiling on all Alder/Raptor/Arrow Lake. AVX-512 is architecturally
  disabled on Arrow Lake; disabled by microcode on Raptor. No current Intel consumer
  CPU has usable AVX-512 — Xeon only.
- `movdir64b` is present on i9-14900K and Core Ultra 9 285K; absent on i7-8700K.
- The AVX2 popcount tier (VPSHUFB trick) for AA coverage is implemented
  (`popcnt_aa_row_avx2`, `aa_coverage_span_avx2`). All Intel consumer CPUs now
  have a vectorised path.

**ARM / aarch64**
- NEON is mandatory on all ARMv8-A. No runtime detection needed — `#[cfg(target_arch = "aarch64")]` is sufficient.
- `vcntq_u8` is hardware-native on all targets (M1–M4, Cortex-A72, Cortex-A76).
  Never use `__builtin_popcountll` on ARM — it compiles to a software sequence.
- `STNP` is an allocation hint only, not a cache-bypass guarantee. For zero-fill use
  `DC ZVA` (zeros full cacheline without read). For non-zero solid fill: `vst1q_u8`.
- Apple M4 has SVE2 (128-bit fixed width). An SVE2 popcount tier is implemented
  behind the `nightly-sve2` feature (`svcnt_u8_z` + `svadd_u8_z`). At 128-bit
  width it matches NEON throughput; on wide-SVE2 server chips (Graviton4 full
  width) it gives up to 4× NEON throughput.

### iGPU / integrated compute

**Intel iGPU (UHD/Iris/Arc laptop)**
VA-API JPEG decode is the only operation worth targeting. It falls out of the
Phase C Intel Arc work for free — same API, no extra work. Transfer overhead makes
other compute (ICC, fill) not worth it on non-UMA x86.

**Apple Silicon (M1–M4)**
Unified Memory Architecture eliminates PCIe transfer overhead. Metal Compute for
ICC CLUT lookup and AA fill is worth measuring; VideoToolbox for JPEG decode is a
zero-copy path. This is a separate work stream (macOS feature flag, `metal` crate).

**Raspberry Pi 4/5**
VideoCore GPU is not accessible as a general-purpose compute API via a stable
userspace interface. NEON is the correct answer on Pi.

Full platform roadmap: `ROADMAP_INTEL.md`.

---

## 6. Concurrency model

```
main thread
  │
  ├── rayon ThreadPool (N workers, default = logical cores)
  │     │
  │     ├── worker 0: render page N   ─── TLS NvJpegDecoder (lazy init)
  │     ├── worker 1: render page N+1 ─── TLS NvJpegDecoder
  │     └── ...
  │
  │   pool.install() blocks until all pages done
  │
  ├── pool.broadcast(release_gpu_decoders)   ← explicit teardown while driver live
  │
  └── pool dropped
```

**GPU decoder lifecycle**
Each rayon worker thread lazily initialises a `NvJpegDecoder` on first use and
stores it in thread-local storage. At process exit all TLS destructors fire
concurrently, which races with the CUDA driver's own atexit shutdown. The fix:
call `release_gpu_decoders()` via `pool.broadcast()` after `pool.install()` returns.
This drops each decoder while the driver is still fully live; the TLS destructors
see `Uninitialised` and are no-ops.

**Parallel fill** (feature: `rayon`)
`fill_parallel` splits the target bitmap into horizontal bands. Each band is a
`BitmapBand` (`&mut` slice of rows). Bands are independent — no synchronisation.
Threshold: `PARALLEL_FILL_MIN_HEIGHT`.

**Font engine locking**
`FontEngine` is behind `Arc<Mutex<_>>`. The mutex is held only during FreeType
face initialisation (rare). Glyph rendering is lock-free via `DashMap<GlyphKey, GlyphBitmap>`.

---

## 7. Error model

**Per-document errors** — abort the whole document:
- `InterpError::Pdf` — lopdf parse failure
- `InterpError::JavaScript` — rejected at open time

**Per-page errors** — skip the page, continue rendering:
- `InterpError::PageOutOfRange`
- `InterpError::MissingResource`
- `InterpError::InvalidPageGeometry`
- `RasterError::PageDegenerate` / `RasterError::PageTooLarge`

**Silent degradation** — never exposed to the caller:
- Unrecognised PDF operators — discarded
- GPU unavailable / below threshold — CPU path used
- Unsupported image transform (`ArbitraryTransformSkipped`) — image skipped; rest of page renders

**Invariant violations** — panic (not user-triggerable):
- CUDA returns success but gives null handle — driver bug only
- `OUT_DIR` missing during build — cargo invariant

---

## 8. Key design decisions

**No subprocess, no Leptonica.**
Everything runs in-process. The Poppler wrapper (`pdf_bridge`) is kept as a
reference baseline for pixel-diff regression tests; it is never linked in the
production render path.

**Generic over pixel type.**
`Bitmap<P>` and all fill functions are generic over `P: Pixel`. The compiler
monomorphises one code path per pixel format. This eliminates runtime mode
dispatch in hot loops at the cost of slightly larger binaries.

**Lazy resource resolution.**
`PageResources` holds an immutable `&Document`. Fonts, images, colour spaces,
and shadings are resolved on first access. No per-page pre-computation pass.

**Compositing pipeline is selected, not branched.**
`PipeState::is_simple_path()` selects one of three monomorphised pipeline variants
before the fill loop starts. The inner loop has no mode branches.

**GPU is additive, not required.**
Every GPU function has a CPU counterpart. Feature flags gate the GPU dep entirely.
A `cargo build` without any `gpu-*` feature produces a fully functional binary that
requires no CUDA toolkit and links no NVIDIA libraries.

**One GPU context per process.**
`GpuCtx` is `Arc<_>`, initialised once, shared by all worker threads. PTX kernels
are loaded at init time. The driver serialises concurrent CUDA calls internally.

**SIMD dispatch is per-arch, not per-feature.**
ARM NEON is unconditionally available on aarch64 — no runtime probe. x86 SIMD
uses `is_x86_feature_detected!` at runtime. Each arch gets a private `dispatch_*`
function; there is no single god-function with cfg-gated returns.
