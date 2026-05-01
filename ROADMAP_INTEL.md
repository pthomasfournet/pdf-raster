# Cross-platform roadmap — CPU SIMD, VA-API GPU, ARM, Apple Silicon

Tracking work needed to run pdf-raster on Intel x86-64, AMD iGPU, ARM, and Apple
Silicon. The core raster pipeline is portable; the work is auditing, fixing build
issues, adding SIMD fast paths, and adding VA-API GPU acceleration.

**Current priorities:** Phase C2 (VA-API JPEG decode) ✓ complete. Next: C3 evaluation
(VA-API VPP deskew — verdict already recorded: not viable), then ARM NEON (Phase E,
already complete). Apple Silicon (Phase F) requires macOS hardware.

---

## Test machine (192.168.1.185 — `tom-Ubuntu`)

Confirmed hardware profile (Apr 2026):

| | |
|---|---|
| **CPU** | Intel Core i7-8700K (Coffee Lake, 2017) — 6C/12T, 3.7 GHz base / 4.7 GHz boost |
| **SIMD** | AVX2, SSE4.1/4.2, BMI1/2, POPCNT — **no AVX-512**, no movdir64b, no VNNI |
| **RAM** | 32 GB DDR4, ~25 GB available |
| **GPU** | RTX 2080 Super (TU104, Turing CC 7.5), 8 GB GDDR6 |
| **Driver** | 570.195.03 |
| **CUDA** | 12.4 (toolkit), `libnvjpeg.so.12` at `/usr/lib/x86_64-linux-gnu/` |
| **OS** | Ubuntu 25.10, kernel 6.17.0-20 |
| **Disk** | 221 GB free |

Notable differences from the Ryzen dev machine:
- **No AVX-512** — all `avx512*` paths fall back to AVX2/scalar at runtime.
  Representative of the entire Intel consumer desktop line: AVX-512 is disabled
  on all Alder/Raptor/Arrow Lake CPUs (microcode on 14900K, architectural on
  285K). AVX2 is the ceiling and will remain so until AVX10.2 (Nova Lake, TBD).
- **No movdir64b** — the i7-8700K (Coffee Lake, 2017) predates movdir64b; the
  non-temporal fill path falls back to AVX2 stores. Newer Intel CPUs (i9-14900K,
  Core Ultra 9 285K) do have movdir64b, so the test bench is conservative here —
  the AVX2 fallback will be tested even if production hardware would use movdir64b.
- **No 3D V-Cache** — L3 is 12 MB vs 128 MB; large edge tables will spill to DRAM.
- **Turing GPU** — `NVJPEG_BACKEND_HARDWARE` is **not available** on Turing (CC 7.5).
  The docs list support as Ampere (A100/A30), Hopper, Ada, Blackwell, and Jetson Thor only.
  We use `GPU_HYBRID` regardless, so this is moot — no code change required.
- **CUDA 12.4** — PTX compiled for `sm_120` will not run; build with
  `CUDA_ARCH=sm_75` on this machine.

---

## SIMD landscape reference

| Feature | Ryzen 9 9900X3D | Intel i7-8700K (test bench) | Intel i9-14900K / Ultra 9 285K | ARM NEON (ARMv8) | ARM SVE2 (Graviton4) |
|---|---|---|---|---|---|
| AVX2 | ✓ | ✓ | ✓ | — | — |
| AVX-512F/BW/VL | ✓ | — | — (disabled in µcode / arch) | — | — |
| AVX-512BITALG | ✓ | — | — | — | — |
| AVX-512VPOPCNTDQ | ✓ | — | — | — | — |
| movdir64b | — | — | ✓ (present on both) | — | — |
| POPCNT (scalar 64-bit) | ✓ | ✓ | ✓ | ✓ (ARMv8.1+) | ✓ |
| NEON VCNT (byte popcount) | — | — | — | ✓ | ✓ |
| SVE2 VCNT (vectorised) | — | — | — | — | ✓ |
| Non-temporal stores | MOVNTI/AVX-512 | MOVNTI | MOVNTI + movdir64b | `DC ZVA` (limited) | `STNT1` |

The i7-8700K test bench is representative of the entire Intel consumer line for SIMD purposes —
AVX2 is the ceiling on all of them. The i9-14900K and Core Ultra 9 285K add movdir64b (which
activates the non-temporal fill path) but are otherwise identical for our code paths. Neither
has usable AVX-512: the 14900K's is disabled by microcode, the 285K's by architectural design
(E-cores can't support it). No current Intel consumer CPU has AVX-512; Xeon is the only path.

**Popcount strategy by platform:**
- **Ryzen**: `avx512bitalg` → `avx512vpopcntdq` → scalar
- **Intel (all consumer)**: scalar `NIBBLE_POP` table (AVX2 VPSHUFB tier to be added — see A2)
- **ARM NEON**: `VCNT` (8-bit per byte) + `VPADAL` reduction → sum across 128-bit register
- **ARM SVE2**: same as NEON but variable-width; ~3× faster at scale (libpopcnt benchmarks on Graviton3/4)

---

## Phase A — CPU fallback audit and fixes

Everything that has a GPU fast path also has a CPU fallback. These fallbacks were
written and tested on AVX-512 hardware; they need an explicit pass on AVX2-only Intel.

### A1 — nvJPEG fallback (DCTDecode) ✓ DONE

When `nvjpeg` feature is disabled, DCTDecode goes through `zune-jpeg` (pure Rust).
No work needed — this path has always been the default.

### A2 — AVX2 AA popcount tier

`aa_coverage_span` dispatch in `crates/raster/src/simd/popcnt.rs`:
1. `avx512bitalg` + `avx512bw` — 128 px/iter (Ryzen only)
2. `avx512vpopcntdq` + `avx512bw` — 64 px/iter (Ryzen only)
3. Scalar `NIBBLE_POP` table — **only tier available on Intel consumer / ARM**

Add tier 2.5 between AVX-512 and scalar:

```rust
// AVX2: _mm256_shuffle_epi8 as 16-entry SIMD lookup table (VPSHUFB popcount trick)
// 32 bytes per iter, ~4× faster than scalar
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn aa_coverage_span_avx2(aabuf: &AaBuf, ...) -> ...
```

Runtime dispatch: `is_x86_feature_detected!("avx2")` before AVX-512 check.
Same trick applies for `popcnt_aa_row` (add an AVX2 tier there too).

**For ARM**: add a NEON tier using `vcntq_u8` + `vpaddlq_u8` + `vaddvq_u16`.
Gate with `#[cfg(target_arch = "aarch64")]` + `#[target_feature(enable = "neon")]`.

### A3 — Tile fill fallback

Subsumed by A2 — both use the same AA coverage path. No separate work.

### A4 — AVX2 ICC CMYK→RGB tier ✓ DONE (Apr 2026)

`crates/gpu/src/cmyk.rs`: dispatch is AVX-512 (16px) → AVX2 (8px) → scalar.

**Pre-existing bug fixed (Apr 2026, E2):** the AVX-512 path was previously gated
with `#[cfg(target_feature = "avx512f")]` (compile-time), which compiled out the
scalar fallback on native Ryzen builds — causing SIGILL on any Intel machine that
lacks AVX-512. Fixed to use `is_x86_feature_detected!()` at runtime; the scalar
path is now always compiled in.

`cmyk_to_rgb_avx2()`: scalar gather AoS→SoA (AVX2 VPSHUFB is lane-local),
`_mm_loadl_epi64` + `_mm256_cvtepu8_epi16` to load 8 u8 → 8 u16,
`_mm256_mullo_epi16` for products, div255 identity `(n+(n>>8)+1)>>8`,
`packus+permute4x64(0x88)+storel_epi64` to narrow back to u8.

**ARM NEON (E2) ✓ DONE:** `cmyk_to_rgb_neon()` added — `vmull_u8` + `vshrn_n_u16`,
8px/iter, per-arch `dispatch_cmyk_matrix()` helper.

### A5 — `movdir64b` solid fill fallback verification

`blend_solid_rgb8` in `crates/raster/src/simd/blend.rs`: dispatches
`movdir64b` → AVX2 → scalar. On Intel the AVX2 path fires. The `movdir64b`
bypass-L3 motivation (preserving 3D V-Cache residency) doesn't apply on Intel;
the AVX2 path is correct and sufficient.

No code change needed. Add a unit test that directly calls `blend_solid_rgb8_avx2`
to ensure the function compiles and produces correct output independently of the
dispatch logic.

### A7 — Intel RDT cache partitioning (deployment note)

The main ROADMAP.md notes `cat_l3`/`cdp_l3` partitioning via Linux `resctrl` to
reserve a fixed L3 partition for edge tables in a batch/server context. This applies
equally to Intel hardware — Intel invented RDT (Resource Director Technology) and
the `resctrl` interface is identical.

On the i7-8700K test bench: L3 is 12 MB (vs 128 MB 3D V-Cache on Ryzen). At 150 DPI
on A4, the edge table for a complex page is ~2–4 MB; it will frequently spill to DRAM
without partitioning. RDT partitioning on Intel server chips (Xeon Scalable) is more
coarse-grained than on the 9900X3D. On Intel consumer chips (including i7-8700K)
**RDT is not supported** — the `resctrl` filesystem will fail to mount.

**Action:** When documenting deployment for batch/server use, note that L3 partitioning
requires a Xeon server CPU on Intel (not consumer). On AMD Ryzen the same `resctrl`
mount works. No code change; deployment documentation only.

### A6 — CPU bilinear deskew rotation

`rotate_cpu` in `crates/pdf_raster/src/deskew/rotate.rs` is the only path when
`gpu-deskew` feature is off. Already tested for correctness. Performance target
< 5ms for 8.4 MP — likely fine on modern Intel; measure to confirm.

---

## Phase B — Build system ✓ DONE

### B1 — PTX compilation guard in build.rs ✓ FIXED (Apr 2026)

The NVCC kernel compilation loop in `crates/gpu/build.rs` previously ran
unconditionally, requiring NVCC even on CPU-only builds (Intel without CUDA, ARM).

**Fixed**: loop is now gated behind `CARGO_FEATURE_GPU_AA || CARGO_FEATURE_GPU_ICC`.
CPU-only `cargo build --release` no longer requires NVCC or the CUDA toolkit.

### B2 — `libnvjpeg.so` / `libcuda.so` link guards ✓ DONE

The `gpu` crate is an optional dep (`dep:gpu`) activated only by GPU features.
On CPU-only builds it is never compiled and no CUDA `.so` links are emitted.
Confirmed by `cargo check` (no GPU features) completing cleanly.

---

## Phase C — VA-API GPU acceleration (iGPU + Intel Arc / AMD discrete)

VA-API (`libva`) is the unified Linux hardware video decode API. It works on:
- **AMD iGPU** (Raphael/RDNA 2, VCN 4.0) — **confirmed working on dev machine** (`renderD129`, Mesa RadeonSI 25.2.8, VA-API 1.20, Apr 2026)
- **Intel iGPU** (Skylake+ UHD / Iris Xe) — Quick Sync JPEG engine; same `libva` API
- **Intel Arc discrete** — same `libva` API, more compute units
- **AMD discrete** (RX 6000/7000) — VCN block exposed via Mesa RadeonSI

The same `VapiJpegDecoder` Rust implementation serves all of these — no per-vendor code.

**Hardware confirmed available on dev machine (`/dev/dri/renderD129`):**
```
AMD Ryzen 9 9900X3D — Raphael iGPU (RDNA 2, VCN 4.0)
Mesa Gallium 25.2.8 / radeonsi / VA-API 1.20
VAProfileJPEGBaseline:  VAEntrypointVLD   ← JPEG baseline decode ✓
VAProfileNone:          VAEntrypointVideoProc  ← VPP (scale, colour convert, etc.)
```

**No prerequisite hardware needed** — VA-API JPEG decode is implementable and testable
on the dev machine today. The i7-8700K test bench also has Intel UHD 630 via the same API.

### C1 — GPU abstraction traits ✓ DONE (Apr 2026)

`crates/gpu/src/traits.rs` defines:
- `GpuJpegDecoder: Send` — `decode_jpeg(&mut self, data, width, height) → Result<DecodedImage>`
- `GpuJpeg2kDecoder: Send` — `decode_jpeg2k(&mut self, data) → Result<DecodedImage>`
- `GpuCompute: Send + Sync` — `icc_clut(&self, pixels, clut, grid_n, width, height) → Result<()>`
- `GpuDecodeError` — unified error wrapping `Box<dyn Error + Send + Sync>`
- `DecodedImage` — interleaved u8 pixels + dims + component count

`NvJpegDecoder`, `NvJpeg2kDecoder`, and `GpuCtx` implement these traits.
Inline image GPU dispatch is wired (`decode_inline_image` passes GPU params through).

### C2 — VA-API JPEG decoder (`VapiJpegDecoder`) ✓ DONE

Implements `GpuJpegDecoder` using raw `libva`/`libva-drm` FFI (no bindgen — the
VA-API surface is small and stable; same rationale as raw CUDA driver API in nvJPEG).

**Device selection:** open `/dev/dri/renderD129` (AMD iGPU on dev machine) via
`vaGetDisplayDRM`. The device node should be configurable or auto-detected, not hardcoded.

**Key API objects:**
```c
VADisplay  dpy;           // vaGetDisplayDRM(fd)  — fd = open("/dev/dri/renderD129", O_RDWR)
VAConfigID cfg;           // vaCreateConfig(dpy, VAProfileJPEGBaseline, VAEntrypointVLD, NULL, 0, &cfg)
VAContextID ctx;          // vaCreateContext(dpy, cfg, width, height, VA_PROGRESSIVE, NULL, 0, &ctx)
VASurfaceID surface;      // vaCreateSurfaces(dpy, VA_RT_FORMAT_YUV420, width, height, &surf, 1, NULL, 0)
VABufferID  pic_buf;      // vaCreateBuffer — VAPictureParameterBufferType (VAPictureParameterBufferJPEGBaseline)
VABufferID  iq_buf;       // vaCreateBuffer — VAIQMatrixBufferType (VAIQMatrixBufferJPEGBaseline)
VABufferID  slice_buf;    // vaCreateBuffer — VASliceDataBufferType (raw JPEG bitstream bytes)
VABufferID  slice_param;  // vaCreateBuffer — VASliceParameterBufferType (VASliceParameterBufferJPEGBaseline)
```

**Decode sequence:**
1. `vaBeginPicture(dpy, ctx, surface)`
2. `vaRenderPicture(dpy, ctx, [pic_buf, iq_buf, slice_param, slice_buf], 4)`
3. `vaEndPicture(dpy, ctx)`
4. `vaSyncSurface(dpy, surface)` — block until VCN JPEG engine completes
5. `vaDeriveImage` or `vaGetImage` → `vaMapBuffer` → copy NV12 planes to host
6. CPU NV12→RGB8 (BT.601 full-range): Y plane + interleaved UV → interleaved RGB
7. `vaUnmapBuffer`, `vaDestroyImage`, destroy buffers

**Output format — NV12 on Raphael (important):** Raphael is VCN 4.0.0 which is
`RDECODE_JPEG_VER_2` in Mesa. Hardware format-conversion (direct RGB output from
the JPEG engine) requires `RDECODE_JPEG_VER_3`, only available on VCN 4.0.3 or
VCN 5.0+. On VCN 4.0.0 the hardware always outputs NV12 (luma plane +
interleaved UV plane). CPU YUV→RGB conversion is required; with AVX2 it is
~0.3ms for a 4MP image and is not the bottleneck relative to JPEG decode itself.
Future hardware (VCN 4.0.3+, e.g. Strix Point) can skip this step via
`VA_RT_FORMAT_RGB32` surface — the code should branch on the detected VCN version
or query `vaQueryConfigAttributes` for `VAConfigAttribRTFormat` support.

**CMYK:** Not in the VA-API surface model at all; fall through to CPU `zune-jpeg`.
**Progressive JPEG:** baseline only via `VAEntrypointVLD`; fall through to CPU.
**Grayscale:** Try `VA_RT_FORMAT_YUV400` surface; if driver rejects it, fall back
to `VA_RT_FORMAT_YUV420` and use only the Y plane from the derived image.

**Thread safety (confirmed from libva source):** `VADisplay` is shared and
thread-safe. `VAContext` must not be used concurrently from multiple threads —
each Rayon worker thread creates its own `VAContext` + `VASurface` pool. Same
`DecoderInit<T>` thread-local pattern as nvJPEG. Mesa radeonsi also holds an
internal mutex per driver instance, but context-level state is not protected.

**Feature flag:** `gpu/vaapi` + `pdf_interp/vaapi`. Link `libva.so.2` + `libva-drm.so.2`.
Zero-cost when disabled.

**Threshold:** Start at `GPU_JPEG_THRESHOLD_PX` (512×512 = 262 144 px). VA-API context
creation has per-image overhead (~0.5–2ms depending on driver); recalibrate once
implemented by running `threshold_bench` equivalent on `renderD129`.

**Dispatch priority:** when both nvJPEG and VA-API are compiled in, nvJPEG takes
priority (discrete GPU is faster for large images). VA-API fires when nvJPEG is
absent or returns `GpuDecodeError` (e.g. on a machine with only iGPU).

### C3 — VA-API VPP for deskew rotation (evaluate)

`VAEntrypointVideoProc` is confirmed available on `renderD129`. VPP exposes:
- `VAProcFilterNone` pipeline with `VAProcPipelineParameterBuffer`
- `output_region` / `surface_region` for crop+scale
- `rotation_state`: `VA_ROTATION_NONE / 90 / 180 / 270` — **only 90° multiples**

**Arbitrary-angle rotation is not in the VA-API VPP spec.** The `rotation_state` field
only supports 0/90/180/270°. Deskew requires sub-degree precision (e.g. 0.3°).
**Verdict: VA-API VPP cannot do deskew rotation. CPU bilinear or NPP remain the path.**

VPP scaling (`VA_FILTER_SCALING_HQ`) could accelerate the 4× downsample in deskew
angle detection, but the downsample is already fast on CPU (< 0.5ms) and is not a
bottleneck. Not worth the API complexity.

### C4 — BVH winding test on AMD/Intel GPU (evaluate after profiling)

Deferred — see main ROADMAP.md. VCN/Arc don't have dedicated RT cores; Vulkan RT
would be required. Only relevant at O(10 000+) path segments which is not typical.

---

## Phase C2 — iGPU / UMA platform analysis

### AMD iGPU (Raphael/RDNA 2 on Ryzen 9 9900X3D) — confirmed hardware

| Operation | API | Verdict |
|---|---|---|
| JPEG decode | VA-API VCN 4.0 (`renderD129`) | **Worth it — implement as C2.** Shares `VapiJpegDecoder` impl with Intel. |
| General compute (ICC CLUT, AA fill) | Vulkan compute / OpenCL | **Not worth it.** Discrete RTX 5070 handles all GPU compute. iGPU adds no value when discrete is present. |
| Bilinear deskew | VA-API VPP (rotation_state) | **No** — VPP only supports 90° multiples; arbitrary angle not available. CPU path wins. |
| Solid fill | CPU `movdir64b` / AVX2 | CPU wins — bandwidth-limited. |

**Key finding:** the AMD iGPU is strictly a JPEG decode acceleration path on this machine.
All other compute stays on the RTX 5070 (CUDA) or CPU (AVX-512).
The `VapiJpegDecoder` on `renderD129` serves as a fallback decode path when the
discrete GPU is busy or unavailable, and as the primary path on machines without
a discrete CUDA GPU.

### Intel iGPU (UHD 630 / Iris Xe / Arc laptop)

| Operation | API | Verdict |
|---|---|---|
| JPEG decode | VA-API Quick Sync | **Worth it** — same `VapiJpegDecoder`, no extra work. |
| General compute | OpenCL / Vulkan compute | Not worth it vs AVX2 on non-UMA. |
| Bilinear deskew | VA-API VPP | No — 90° only (see C3). CPU wins. |

### Apple Silicon (M1–M4) — UMA changes the calculus (Phase F, macOS only)

| Operation | API | Verdict |
|---|---|---|
| JPEG decode | VideoToolbox (`VTDecompressSession`) | **Worth it.** Zero-copy via CVPixelBuffer. |
| ICC CLUT lookup | Metal Compute | **Worth measuring** — no PCIe penalty; 500K px threshold may drop. |
| AA fill / tile fill | Metal Compute | **Worth measuring** — same UMA argument. |
| Bilinear deskew | Metal Performance Shaders (`MPSImageBilinearScale`) | **Worth it** — MPS supports arbitrary angle. |
| Solid fill | CPU `vst1q_u8` | CPU wins — bandwidth-limited. |

Metal/VideoToolbox require a separate `metal` feature flag and macOS build target.
Not implementable on Linux. Phase F.

### Raspberry Pi 4/5 (VideoCore VI/VII)

VideoCore compute is not accessible for general workloads; hardware JPEG decode is
ISP-only (`libcamera`), not usable for arbitrary bitstreams. **Skip.**
CPU NEON paths (Phase E) are the correct answer.

### Summary

| Platform | JPEG decode | Compute (ICC/fill) | Deskew | Priority |
|---|---|---|---|---|
| AMD iGPU (Raphael, dev machine) | VA-API VCN — `VapiJpegDecoder` | discrete GPU handles it | CPU (VPP = 90° only) | **C2 — implement now** |
| Intel iGPU (UHD/Iris/Arc laptop) | VA-API Quick Sync — same impl | not worth it (non-UMA) | CPU (VPP = 90° only) | C2 — falls out for free |
| Apple M1–M4 | VideoToolbox — zero copy | Metal Compute (UMA) | MPS bilinear (arbitrary angle) | Phase F (macOS) |
| Raspberry Pi 4/5 | not accessible | not worth it | CPU NEON | skip |

---

## Deployment notes — L3 cache partitioning

The main ROADMAP.md lists `cat_l3`/`cdp_l3` partitioning (Linux `resctrl`) as a
deployment note for batch/server use: reserving a fixed L3 slice for edge tables
prevents scan-heavy page images from evicting them during rasterisation.

### Support by platform

| Platform | RDT support | `resctrl` | Notes |
|---|---|---|---|
| **AMD Ryzen 9 9900X3D** | No (AMD PSF, not RDT) | N/A | 128 MB 3D V-Cache makes partitioning unnecessary — working set fits entirely in L3 |
| **Intel Xeon Scalable (Skylake-SP+)** | Yes — CAT + CDP | `mount -t resctrl resctrl /sys/fs/resctrl` | 2–4 MB L3 allocation units (CLOS); `cat_l3` reserves slices for edge tables |
| **Intel consumer (i7-8700K, 14900K, Ultra 285K)** | **No** | will fail to mount | RDT is a server/Xeon feature; absent from all consumer Intel parts |
| **ARM server (Graviton4, Ampere Altra)** | MPAM (ARM equivalent) | kernel 5.18+ `resctrl` via MPAM driver | Similar concept; MPAM partitions are cache way masks like RDT |

### Practical guidance

For the **i7-8700K test bench** (12 MB L3, no RDT): at 150 DPI on A4 the edge
table for a complex page is 2–4 MB. With a concurrent image decode filling the
remaining 8–10 MB, edge table eviction is possible on dense-vector pages. No
software mitigation available — this is a hardware ceiling. Measure with `perf
stat -e LLC-load-misses` if throughput regresses vs the Ryzen machine.

For **production batch servers**, prefer Intel Xeon or AMD EPYC (both support
cache partitioning via `resctrl`). If deploying on Intel Xeon:

```bash
# Mount resctrl (requires kernel CONFIG_X86_CPU_RESCTRL=y)
mount -t resctrl resctrl /sys/fs/resctrl

# Check available L3 capacity bitmask (example: 0x3ff = 10 ways)
cat /sys/fs/resctrl/info/L3/cbm_mask

# Reserve ways 0-3 (0x00f) for the rasteriser process group
mkdir /sys/fs/resctrl/pdf_raster
echo "L3:0=0x00f" > /sys/fs/resctrl/pdf_raster/schemata
echo $$ > /sys/fs/resctrl/pdf_raster/tasks
```

Tune the bitmask to the actual edge table working set size; over-reserving wastes
capacity for page image decode. The optimal split is document-class dependent.

### No code change required

All rasteriser code already runs correctly without partitioning — this is a
deployment tuning option for operators running pdf-raster in a multi-tenant batch
server context. No feature flag, no build change.

---

## Phase D — Portability CI ✓ DONE (Apr 2026)

`.github/workflows/ci.yml` has a `cpu-only` job that runs on every push/PR:

- `cargo build --release -p cli` — no GPU features, no CUDA toolkit required
- `cargo test -p pdf_interp --lib` — resource decoding, CPU paths
- `cargo test -p raster --lib` — full SIMD dispatch, fill, blend, popcnt suite
- `cargo test -p gpu --lib -- cmyk` — AVX2 and scalar CMYK parity
- `cargo test -p pdf_raster --lib` — deskew, render helpers
- `cargo clippy -p pdf_interp -p raster -p gpu -p pdf_raster -- -D warnings`

Runner is `ubuntu-latest` (x86-64, AVX2, no AVX-512) — representative of the
entire Intel consumer line. AVX2 and scalar fallback tiers are exercised on every
push. A `fmt` job runs `cargo fmt --all -- --check` in parallel.

---

## Phase E — ARM NEON (next CPU target)

The next CPU after Intel is ARM. All target platforms share a common safe NEON
baseline; platform-specific extensions are gated separately.

### Target platform matrix

| Platform | Core | Arch | NEON | vcntq_u8 | vmull_u8 | STNP | FP16 | Dot | Notes |
|---|---|---|---|---|---|---|---|---|---|
| **Apple M1** | Firestorm | ARMv8.6-A | ✓ | ✓ | ✓ | hint only | conv only | ✓ | No SVE |
| **Apple M2** | Avalanche | ARMv8.6-A | ✓ | ✓ | ✓ | hint only | ✓ | ✓ | No SVE |
| **Apple M3** | Everest | ARMv8.6-A | ✓ | ✓ | ✓ | hint only | ✓ | ✓ | No SVE |
| **Apple M4** | Palma | ARMv9.2-A | ✓ | ✓ | ✓ | hint only | ✓ | ✓ | SVE (128-bit) |
| **Raspberry Pi 4** | Cortex-A72 | ARMv8.0-A | ✓ | ✓ | ✓ | hint only | ✗ | ✗ | No crypto in BCM2711 |
| **Raspberry Pi 5** | Cortex-A76 | ARMv8.2-A | ✓ | ✓ | ✓ | hint only | ✓ | ✓ | 2×256-bit NEON pipeline |

**Safe baseline for ALL six platforms** (write once, runs everywhere):
```rust
vcntq_u8()               // popcount — hardware native, not emulated
vpaddlq_u8() / vaddvq_u16() // reduction
vmull_u8() / vshrn_n_u16()  // u8×u8 → u16 → u8 (CMYK math)
vbslq_u8()               // bitwise select (glyph unpack)
vld1q_u8() / vst1q_u8()  // 128-bit load/store (solid fill)
vmlaq_f32()              // f32 multiply-accumulate (bilinear sampling)
```

### Important findings

**`vcntq_u8` is hardware-native on all targets** — not emulated. This is the
foundation of the ARM AA coverage path. The pattern:
```rust
let cnt = vcntq_u8(v);           // popcount each byte → 16 bytes of counts
let sum16 = vpaddlq_u8(cnt);     // pairwise add → 8 × u16
let total = vaddvq_u16(sum16);   // horizontal sum → scalar u16
```
processes 128 bits of coverage mask in ~3 instructions.

**No scalar POPCNT instruction on ARM** — `__builtin_popcountll` compiles to a
software sequence. Always use `vcntq_u8` for popcount-heavy paths. This is
different from x86 where scalar `POPCNT` is a single-cycle instruction.

**Non-temporal stores are hints only** — `STNP` (Store Non-Temporal Pair) is an
allocation hint, not a cache-bypass guarantee like x86 `MOVNTI`. For zero-fill,
`DC ZVA` (Data Cache Zero by VA) is more effective: zeroes a full cacheline
without reading it first. Block size is in `CTR_EL0`/`DCZID_EL0` (typically 64
bytes). For non-zero solid fills, regular `vst1q_u8` is the correct approach —
no streaming store equivalent exists in baseline NEON.

**Raspberry Pi 4 has no crypto** — BCM2711 did not license the Cortex-A72 crypto
extension. Do not use AES/SHA intrinsics on RP4 without a runtime check. Not
relevant to the rasterizer (we don't use crypto), but worth noting for the build.

**Apple M4 has SVE (128-bit fixed width)** — not the variable-width SVE of server
chips. Effectively a 128-bit NEON alias with SVE instructions. No advantage over
NEON for our workloads; design the NEON tier, it runs unchanged on M4.

### What needs adding to the codebase

| Path | File | Current | ARM NEON equivalent | Gate | Status |
|---|---|---|---|---|---|
| AA row popcount | `raster/src/simd/popcnt.rs` | AVX-512 → `popcnt` → scalar | `vcntq_u8` + `vpaddlq_u8` + `vaddvq_u16`, 16B/iter | `#[cfg(target_arch = "aarch64")]` | ✓ DONE (Apr 2026) |
| AA coverage span | `raster/src/simd/popcnt.rs` | AVX-512 BITALG → scalar NIBBLE_POP | `vcntq_u8` on isolated nibbles, `vst2q_u8` interleave, 32px/iter | same | ✓ DONE (Apr 2026) |
| ICC CMYK→RGB | `gpu/src/cmyk.rs` | AVX-512 → scalar | `vmull_u8` + `vshrn_n_u16`, 8px/iter | same | ✓ DONE (Apr 2026) |
| Glyph unpack (1bpp→8bpp) | `raster/src/simd/glyph_unpack.rs` | SSE2 `_mm_cmpeq_epi8` | `vtstq_u8` test-bits (1 instr vs SSE2's 3), 16px/iter | same | ✓ DONE (Apr 2026) |
| Solid fill | `raster/src/simd/blend.rs` | `movdir64b` → AVX2 → scalar | `vst3q_u8` 16px/iter (RGB); `vst1q_u8` 16px/iter (gray) | same | ✓ DONE (Apr 2026) |
| Bilinear deskew rotation | `pdf_raster/src/deskew/rotate.rs` | AVX2 inner loop | `vmlaq_f32` 4px/iter | same | ✓ DONE (Apr 2026) |

### Dispatch architecture (adopted Apr 2026)

Each public SIMD function delegates to a private per-arch `dispatch_*` helper.
This avoids `return`/cfg fallthrough chains and keeps each arch path in its own
function that the compiler can reason about independently.

```rust
// Pattern used in popcnt.rs — extend to other paths:
pub fn popcnt_aa_row(row: &[u8]) -> u32 { dispatch_popcnt(row) }

#[cfg(target_arch = "x86_64")]
fn dispatch_popcnt(row: &[u8]) -> u32 { /* avx512 → popcnt → scalar */ }

#[cfg(target_arch = "aarch64")]
fn dispatch_popcnt(row: &[u8]) -> u32 { unsafe { popcnt_aa_row_neon(row) } }

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn dispatch_popcnt(row: &[u8]) -> u32 { popcnt_aa_row_scalar(row) }
```

This pattern is the standard for all subsequent NEON work — no god-file dispatch
blocks, each arch gets its own clean function.

### NEON implementation details (AA paths, Apr 2026)

**`popcnt_aa_row_neon`** (16B/iter):
```rust
let v = vld1q_u8(ptr);
let cnt = vcntq_u8(v);           // hardware popcount per byte
let wide = vpaddlq_u8(cnt);      // pairwise widen u8→u16 (avoids overflow)
total += vaddvq_u16(wide) as u32; // horizontal sum
```

**`aa_coverage_span_neon`** (32 output pixels/iter):
```rust
let mask_lo = vdupq_n_u8(0x0F);
let hi = vandq_u8(vshrq_n_u8(v, 4), mask_lo);  // high nibble → bits 3–0
let lo = vandq_u8(v, mask_lo);                   // low nibble
acc_hi = vaddq_u8(acc_hi, vcntq_u8(hi));
acc_lo = vaddq_u8(acc_lo, vcntq_u8(lo));
// after 4 rows: vst2q_u8 interleaves even/odd pixel counts into shape
vst2q_u8(shape.as_mut_ptr(), uint8x16x2_t(acc_hi, acc_lo));
```

### Build changes needed for ARM

- `CUDA_LIB_DIRS` in `build.rs` is x86-64 path-hardcoded — fine, no GPU features
  will be active on ARM.
- All x86 intrinsic functions are already inside `#[cfg(target_arch = "x86_64")]`
  — they compile out cleanly on aarch64. **Verified** with:
  ```bash
  cargo check --target aarch64-unknown-linux-gnu  # ✓ clean (Apr 2026)
  ```
- `is_x86_feature_detected!` is x86-only. ARM dispatch uses `#[cfg(target_arch = "aarch64")]`
  directly — no runtime detection macro needed. NEON is mandatory on all ARMv8-A
  (all six target platforms), so the `#[cfg]` is both necessary and sufficient.

### SVE2 (Graviton4, Ampere Altra) ✓ DONE (Apr 2026)

`popcnt_aa_row_sve2` and `aa_coverage_span_sve2` are implemented in
`crates/raster/src/simd/popcnt.rs` behind the `nightly-sve2` Cargo feature.
Dispatch is above NEON: `is_aarch64_feature_detected!("sve2")` at runtime.

Key intrinsics used: `svld1_u8` (predicated load), `svlsr_u8_z` (logical shift
right), `svand_u8_z` (AND with predicate), `svcnt_u8_z` (byte popcount),
`svadd_u8_z` (add), `svaddv_u8` (horizontal sum), `svst1_u8` (store).

On fixed-128-bit SVE2 (Apple M4, Graviton4 128b mode): `svcntb() == 16`,
same throughput as NEON. On wide-SVE2 server chips (Graviton4 full width):
`svcntb()` up to 64, giving up to 4× NEON throughput.

`cargo check` is clean on stable (feature off) and nightly (feature on).
Parity tests `sve2_popcnt_matches_scalar` and `sve2_coverage_matches_scalar`
skip at runtime when `sve2` is not detected — they will execute on Graviton4.

---

## Summary checklist

| Item | Priority | Status |
|---|---|---|
| B1 — build.rs NVCC guard | high | ✓ DONE (Apr 2026) |
| B2 — linker lib guard | high | ✓ DONE (confirmed) |
| A1 — nvJPEG fallback (zune-jpeg) | low | ✓ DONE (always worked) |
| E4 — `cargo check` aarch64 clean | high | ✓ DONE (Apr 2026) |
| E1 — ARM NEON AA row popcount | high | ✓ DONE (Apr 2026) |
| E1 — ARM NEON AA coverage span | high | ✓ DONE (Apr 2026) |
| A2 — AVX2 AA popcount tier | medium | ✓ DONE (Apr 2026) — VPSHUFB nibble LUT, popcnt_aa_row 32B/iter, aa_coverage_span 64px/iter |
| A4 — AVX2 ICC CMYK tier | medium | ✓ DONE (Apr 2026) — `_mm256_mullo_epi16` 8px/iter, div255 identity, scalar gather AoS→SoA |
| A5 — movdir64b fallback unit test | low | ✓ DONE (Apr 2026) — `avx2_rgb8_matches_scalar_direct` + `avx2_gray8_matches_scalar_direct` in blend.rs |
| A6 — CPU bilinear deskew perf | low | ✓ DONE (Apr 2026) — `rotate_cpu_8mp_timing` #[ignore] test added; scalar loop LLVM auto-vectorises; Ryzen baseline ~0.6 ms |
| E2 — ARM NEON ICC CMYK→RGB | medium | ✓ DONE (Apr 2026) — also fixed AVX-512 compile-time dispatch bug |
| E3 — ARM NEON glyph unpack | low | ✓ DONE (Apr 2026) — `vtstq_u8` replaces SSE2's `and+cmpeq+xor` |
| E6 — ARM NEON solid fill | low | ✓ DONE (Apr 2026) — `vst3q_u8` RGB + `vst1q_u8` gray, per-arch dispatch helpers |
| E7 — ARM NEON bilinear deskew | low | ✓ DONE (Apr 2026) — `vmlaq_f32` 4px/iter, OOB lanes scalar fallback, per-arch dispatch |
| D — CI CPU-only job | medium | ✓ DONE (Apr 2026) — `.github/workflows/ci.yml` |
| A7 — Intel RDT cache partitioning note | low | ✓ DONE — see "Deployment notes" section above |
| C1 — `GpuJpegDecoder` / `GpuCompute` abstraction traits | low | ✓ DONE (Apr 2026) — `crates/gpu/src/traits.rs`; `NvJpegDecoder` + `NvJpeg2kDecoder` + `GpuCtx` impls; inline image GPU gap also closed |
| C2 — `VapiJpegDecoder` (VA-API JPEG decode) | **high** | **UNBLOCKED** — AMD iGPU confirmed on dev machine (`renderD129`, VCN 4.0, VA-API 1.20, Apr 2026); design complete; implement next |
| C3 — VA-API VPP deskew rotation | low | Not feasible — `rotation_state` supports 90° multiples only; arbitrary-angle deskew stays on CPU/NPP |
| C4 — BVH winding test (AMD/Intel RT via Vulkan) | low | design noted; implement only if profiling shows complex-path bottleneck |
| E5 — SVE2 popcount tier | low | ✓ DONE (Apr 2026) — `nightly-sve2` Cargo feature; `popcnt_aa_row_sve2` + `aa_coverage_span_sve2`; `svlsr_u8_z` + `svcnt_u8_z` + `svadd_u8_z`; dispatch above NEON; `cargo check` clean on stable (feature off) and nightly (feature on); runs when `is_aarch64_feature_detected!("sve2")` returns true |
| aarch64 CI job | medium | ✓ DONE (Apr 2026) — `aarch64-check` job in `.github/workflows/ci.yml`; stable NEON + nightly SVE2; installs `gcc-aarch64-linux-gnu` for C dep crates |
