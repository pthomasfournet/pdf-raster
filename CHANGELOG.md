# Changelog

All notable changes to this project will be documented in this file.

## [0.8.0] - 2026-05-09

### Bug Fixes

- Async disk writer + opt-in tier + cold-start lookup
- Review pass — page_h to f32, doc accuracy, IccClut clut required
- Move cuda_backend_smoke imports inside cfg-gated fn
- Silence underscore-binding lint on cuCtxGetApiVersion probe
- Clear -D warnings clippy across feature matrix
- Clear -D warnings clippy across feature matrix
- Probe CUDA + nvjpeg2k locations instead of pinning cuda-12.8
- Hardening pass on Phase 10 task 2
- Hardening pass on Vulkan backend

### Chores

- Drop stale CUDA-12.8 path pins after driver/toolkit bump

### Documentation

- Backfill v0.7.0 release notes
- Document why DevicePageBuffer / DeviceImageCache stay un-generified
- Mark Phase 10 task 1 partial — trait + CudaBackend shipped
- Phase 10 Task 3 follow-ups + scrub stale comments
- Phase 10 Task 4 — renderer migration + bench gate

### Features

- Scaffold GpuBackend trait + Params
- CudaBackend init + alloc + budget; record_* stubbed
- Wire CudaBackend record_* + submit_page/wait_page
- Phase 10 task 2 — Slang port of all kernels
- Phase 10 task 3 — Vulkan compute backend
- Aa_fill 2D dispatch + blit_image push-constant inv_ctm
- Persist VkPipelineCache to disk across runs
- Reusable staging buffer in Vulkan TransferContext
- Plumb BackendPolicy::ForceVulkan + --backend vulkan
- ForceVulkan errors loudly until renderer migration lands
- Wire VulkanBackend through the renderer + nvcc-probe build fix

### Other

- Add image-cache matrix driver
- Auto-probe CUDA + nvjpeg2k library paths
- Treat nvjpeg2k as optional, not gating
- Full matrix on both reference machines — gate FAILS
- Re-bench after disk-tier rework — gate goes from 14× → 1.1-1.9× regression
- Trait surface — zero-size rejection, VramBudget invariant, state-machine docs
- BlitParams::validate enforces NaN/Inf/zero-dim invariants
- Aa_fill_gpu early-returns on n_pixels == 0
- Drop stale phase/task refs from doc + panic msg
- Cuda_backend_smoke — actionable expect/assert messages
- Reject Mask layout in BlitParams::validate, use be() helper, drop task ref
- BackendError::msg constructor, demote StringError to private
- Phase 10 task 4 step 4 — bench gate, all measurable criteria PASS

### Refactor

- Extract composite_rgba8 into lib_kernels::composite
- Extract apply_soft_mask into lib_kernels::soft_mask
- Extract aa_fill + aa_fill_gpu into lib_kernels::aa
- Extract tile_fill into lib_kernels::tile
- Extract icc_cmyk_to_rgb into lib_kernels::icc
- Re-export blit under lib_kernels for symmetry
- Split mod.rs into budget/eviction/promotion submodules
- Simplify pass on Vulkan backend
- Hardening + simplify pass on the ForceVulkan surface
- Hardening + simplify pass on Vulkan dispatch and build.rs

## [0.7.0] - 2026-05-07

### Bug Fixes

- Set LD_LIBRARY_PATH for any nvjpeg-linked binary
- Drop VA-API from v0.6.0 matrix
- Correct corrupted JITTER_Y Halton(3) values
- Disk tier — split callback errors by side
- RST handling + scaffolding cleanup pass

### Chores

- Scaffold v0.6.0 GPU baseline output directory
- Add clangd config for the CUDA kernels
- Release v0.7.0

### Documentation

- Close Phase 7 bench gate, open Phase 8 (custom on-GPU JPEG)
- Defer Phase 8, open Phase 9 (active) and Phase 10 (planned)
- Update Phase 9 task status through task 4 GPU side
- Mark Phase 9 task 4 done; remaining tasks 5+6
- Explain CUDA_ARCH selection + add cache/feature-flag table

### Features

- Add v0.6.0 baseline aggregation script
- V0.6.0 baseline driver — pre-flight checks
- V0.6.0 baseline driver — build phase
- V0.6.0 baseline driver — bench phase + aggregation
- Jpeg pipeline Phase 0 — CPU pre-pass for self-synchronizing decoder
- Introduce ImageData enum (Phase 9 task 1)
- Phase 9 task 2 — VRAM tier of the device image cache
- Phase 9 task 3 — pinned host RAM demotion tier
- Phase 9 task 4 (GPU side) — image blit kernel + DevicePageBuffer
- Phase 9 task 4 — renderer integration (cache + GPU image blit)
- Phase 9 task 5 — disk tier
- Image-cache prefetcher

### Other

- V0.6.0 GPU baseline matrix — raw results
- Security + correctness pass on Phase 0 CPU pre-pass
- Review pass on ImageData; restore unused_results lint
- Apply review findings on the ImageData hardening pass
- Hardening pass on Phase 9 task 2
- Apply review findings on the hardening pass
- Hardening pass on Phase 9 task 3
- Apply review findings on the host-tier hardening pass
- Review pass on Phase 9 task 4 (GPU side)
- Tighten page_h debug_assert to exact equality
- Review pass on Phase 9 task 4 renderer integration
- Disk tier review pass — single-copy promotes, sharper docs
- Prefetcher review pass
- Prefetcher polish + roadmap close-out

### Performance

- Disable nvJPEG dispatch on consumer Blackwell

### Refactor

- Simplifier sweep across remaining crates (color, raster, font, encode, gpu, pdf_raster, bench)
- V0.6.0 driver as thin wrapper around tests/bench_corpus.sh

## [0.6.0] - 2026-05-07

### Bug Fixes

- Update Send safety comment and fix doc tense in CachedCtx
- Correct misleading comments in create_surface_and_context and test
- Destroy cached context+surface in Drop
- Correct misleading comment in drop_impl_compiles_with_cached_ctx
- Include is_gray in cache key to prevent YUV400/YUV420 mismatch
- Replace pool.install+rayon::scope with pool.scope to fix deadlock
- Spawn n_threads-1 consumers so W0 is free to produce
- Drop tx inside pool.scope to unblock consumers
- Make count_filter const fn to satisfy clippy nursery lint
- Add trailing newline to awk printf to satisfy set -e
- Hardening pass on lazy parser — xref streams, DOS caps, overflow
- Post-rip-out review fixes
- Post-review fixes for --ram (drop dead code, warn on stale dirs)
- PageIter now resolves indirect /Kids references

### Chores

- Untrack ROADMAP_INTEL.md (gitignored)
- Add .worktrees/ to .gitignore
- Release v0.6.0

### Documentation

- Update VapiJpegDecoder and decode_sync docs to reflect context reuse
- Clarify capacity arithmetic in PageQueue doc example
- Update Phase 7 and v0.4.0 with CLI refactor and Rayon hardening
- Audit and update all documentation to v0.5.1
- Mark affinity dispatch complete in Phase 7 work items
- Add version regression history table and bench_versions.sh
- Fix stale lopdf reference in fuzz_ccitt comment
- Update for v0.6.0 (lopdf rip-out, RAM-default output, new bench numbers)
- Regenerate CHANGELOG for v0.6.0

### Features

- Add CachedCtx struct and field to VapiJpegDecoder
- Route JPEG decodes through a single-threaded DecodeQueue
- Bounded work-stealing page queue replaces par_iter
- PageDiagnostics pre-scan pass wires RoutingHint
- Wire affinity dispatch — CpuOnly pages skip GPU decoder init
- Add lazy zero-copy PDF parser crate to replace lopdf
- Wrap dict in Dictionary newtype; add Object::enum_variant
- Rip lopdf out of pdf_interp + pdf_raster, switch to in-tree pdf crate
- Add --ram mode — write output to /dev/shm with dynamic spill-to-disk
- RAM output by default for bare-stem prefixes; --no-ram opts out

### Other

- Wrap long log::warn! line to satisfy rustfmt
- Address all 10-pass review findings
- Fix pool.scope docs, capacity bug, ETA guard, dedup error chain
- Hardening pass on page_queue and main
- Hardening pass — debug_assert, accurate expect reasons, real capacity tests, clearer comments

### Performance

- Reuse VAContext+VASurface across same-dimension decodes
- Gate prescan behind GPU feature flags; no-op on CPU-only builds
- Switch global allocator to mimalloc; add --timings flag
- Pin lopdf to fix commit; add profiling build profile
- Axis-aligned fast path in blit_image inner loop
- Eliminate probe decode in decode_dct CPU path

### Refactor

- Extract create_surface_and_context helper
- Extract DEFAULT_VAAPI_DEVICE const, remove duplicate literals
- Extract diagnostics module from main.rs
- Move build_page_list into Args method, return Result with warnings vec
- Move routing_hint_from_diag+report_progress into page_queue; remove serial prescan
- Extract count_filter+update_max_ppi helpers, remove duplicate PPI code
- Replace PageQueue with par_iter; prescan inline per render thread
- Skip pdftoppm by default; add -R flag to include it
- Split pdftoppm comparison into bench_compare.sh
- Simplify hardened parser — extract dup helpers, normalize accessors
- Simplify --ram wiring (extract encode helper, normalise error style)

### Testing

- Upgrade bench_corpus.sh with hyperfine + mpstat/iostat monitoring

## [0.5.1] - 2026-05-02

### Bug Fixes

- Escalate GPU unexpected-component log to warn

### Chores

- Release v0.5.1

### Documentation

- Mark Phase 7 SOF detection + dispatch refactor complete
- Audit and correct all Phase 7 documentation

### Features

- Add JpegVariant + jpeg_sof_type() peek — shared SOF detection
- Content-aware JPEG dispatch — skip VA-API for progressive JPEG

### Other

- Bump actions/checkout v4 → v6 (Node.js 24)
- Bump actions/cache v4 → v5 (Node.js 24)
- Jpeg_sof — fix None/Other contract, SOS guard, 0xFF prefix check, TEM marker, test coverage
- Jpeg_parser — fix 16-bit DQT, DHT truncation, range validation, SOS/EOI bounds, truncation error

### Refactor

- Remove SOF2 rejection from jpeg_parser — caller owns routing
- Collapse decode_dct_gpu+vaapi into generic decode_dct_gpu_path

### Testing

- Mark sparse-page integration tests #[ignore]

## [0.5.0] - 2026-05-02

### Bug Fixes

- Fix u32 overflow in PageIter; extract should_render; harden render_channel

### Chores

- Fmt and clippy fixes for PageSet feature
- Release v0.5.0

### Documentation

- Update all version references to v0.4.0; add v0.4.0 release entry
- Add render_channel streaming and PageSet sparse-selection examples
- Fix streaming example — remove rayon::scope deadlock risk

### Features

- Add PageSet validated sparse-page-set type
- Add pages field to RasterOptions
- Wire PageSet sparse filtering into render_pages and render_channel

### Refactor

- Harden PageSet — PartialEq/Eq, IntoIterator, safer first/last, edge-case tests
- Harden RasterOptions::pages field — test coverage and comment accuracy

## [0.4.0] - 2026-05-02

### Bug Fixes

- Hardening pass on backend flag implementation
- Resolve clippy warnings under vaapi feature
- Evict PDF from page cache before each timed run
- PTX compilation never triggered on gpu-aa/gpu-icc builds
- Replace infallible expect in col_to_byte with saturating cast
- Remove ncomps param from draw_image/blit_image; derive from P::BYTES
- Propagate FreeType init error instead of panicking
- Correct AA_GAMMA table values and add exhaustive test
- Harden general pipe compositing — 5 bugs, 4 safety assertions
- TJ kern ignores Tz; log path-builder failures
- Minor hardening and log-level fixes

### Chores

- Add plugin runtime directories
- Release v0.4.0

### Documentation

- Add benchmarks.md with full methodology and CPU-only results
- Add VA-API iGPU results + Intel CPU 08 + corpus 09 regression note
- Update all tables with fresh clean-build measurements
- Fresh CPU benchmarks (both machines) + Phase 7 roadmap
- Fresh VA-API corpora 01-05 (uncontested run)
- Add storage type to hardware table, note cold-cache methodology
- Intel GPU results (RTX 2080 Super, Turing sm_75)
- Complete fresh VA-API table (corpora 06-10)

### Features

- Add --backend auto|cpu|cuda|vaapi flag
- Expose vaapi feature flag on CLI crate; correct VA-API benchmark data
- Add --corpus-dir flag for alternate PDF location

### Other

- Fix missing system deps — libfreetype6-dev + bundled FreeType for aarch64
- Install libc6-dev-arm64-cross + LIBZ_SYS_STATIC for cross-compile

### Refactor

- Extract compute_a_src helper; eliminate duplicated alpha logic
- Split page/mod.rs into focused sub-modules
- Simplify pass over review-session changed files
- Extract finish_pixel helper; clarify push_glyph comment

### Testing

- Add hardened corpus benchmark script

## [0.3.0] - 2026-05-01

### Bug Fixes

- Fix three CI failures — rustfmt, SVE2 unsafe blocks, aarch64 dead_code
- Hardening pass on image submodules — 17 bugs fixed
- Hardening pass round 2 — 8 bugs fixed
- Hardening pass — 6 bugs fixed
- 3 correctness bugs + bench hardening

### Chores

- Cargo fmt --all
- Remove unused smallvec dependency
- Remove unused proptest/tempfile dependencies; fix golden tempdir
- Update CHANGELOG.md for v0.3.0
- Release v0.3.0

### Documentation

- Update all docs for v0.2.0 — ARM/aarch64 and VA-API now supported
- Add proptest testing strategy section
- Pre-release documentation update for v0.3.0

### Features

- Add cargo-fuzz targets for CCITTFaxDecode and JBIG2Decode
- Name rayon workers and increase stack size to 8 MiB

### Other

- Cargo fmt

### Performance

- Use Compression::Fast for PNG output
- Cache baked CMYK CLUT tables per page render
- Panic=abort, inline(always) on transfer hot path, black_box bench

### Refactor

- Replace match-with-return-arm with let-else
- Replace #[allow] with #[expect] throughout
- Replace DashMap+lru with quick_cache for glyph cache
- Split 1500-line image/mod.rs into focused submodules

## [0.2.0] - 2026-05-01

### Bug Fixes

- Fix nvJPEG segfault on process exit — eager decoder teardown
- Guard PTX compilation behind GPU feature flags
- Hardening pass — bounds checks, SAFETY docs, dead-code removal
- Correct release.toml schema for cargo-release 1.x

### Chores

- Set GitHub URL and strip email from Cargo metadata
- Set author to Tom in Cargo metadata
- Add versioning tooling — cargo-release + git-cliff
- Gitignore ROADMAP_INTEL.md
- Cargo fmt
- Update CHANGELOG.md and release config for v0.2.0
- Release v0.2.0

### Documentation

- Update performance table with full 10-corpus benchmark results
- Add ARCHITECTURE.md
- Update ROADMAP_INTEL.md for AMD iGPU VA-API discovery
- Mark C2 complete, sync checklist with implemented state

### Features

- Add ARM NEON acceleration for AA popcount paths
- Add NEON for CMYK→RGB and glyph unpack; fix AVX-512 dispatch bug
- Add NEON solid fill for RGB and gray (E6)
- Add NEON bilinear deskew rotation (E7)
- Add AVX2 AA popcount tier (A2)
- Add AVX2 ICC CMYK→RGB tier (A4)
- Add CPU-only CI workflow and fix PTX placeholder generation (D)
- Add SVE2 popcount tier and aarch64 CI job (E5)
- GPU decoder traits + inline image GPU dispatch
- VA-API JPEG decoder for AMD/Intel iGPU on Linux

### Refactor

- Hardening pass on popcnt.rs
- Hardening pass on NEON CMYK, glyph unpack, and popcnt
- Hardening pass on blend.rs
- Hardening pass on cmyk.rs
- Hardening pass on CI workflow and build.rs
- Hardening pass on SVE2 tier and CI fixes
- Hardening pass on traits.rs
- Remove dead hardware_backend field and fix doc accuracy
- Hardening pass on nvjpeg.rs

### Testing

- Add rotate_cpu 8.4 MP timing smoke-test (A6)

## [0.1.0] - 2026-04-30

### CLI

- Add -P/--progress flag for live page-completion feedback
- Wire native renderer behind --native flag
- Hardening pass on native render path
- Remove --native flag; native Rust renderer is now the only path
- Wire GpuCtx into renderer; validate GPU AA quality
- Hardening pass on GPU wiring and error handling
- Wire nvJPEG and nvJPEG2000 decoders per rayon thread
- Hardening pass — fix init-retry spam, DecoderInit state machine, doc fixes
- Remove stale Option<T> thread_local statics left over from hardening pass
- Thin wrapper over pdf_raster; add RasterSession + render_page_rgb
- Use contains() for UserUnit range check
- Fix all workspace warnings

### Chores

- Cargo fmt
- Add Cargo metadata and LICENSE for git dependency use
- Sanitize .gitignore for public GitHub
- Remove private fixture PDFs and sanitize all references
- Sanitize source comments for public release

### Color

- Hardening pass on cmyk_to_rgb_reflectance and gray_to_u8

### Documentation

- Update ROADMAP and CLAUDE.md with calibrated thresholds
- Update ROADMAP and CLAUDE.md with Phase 3 follow-on completions
- Mark Type 1 shading complete in ROADMAP
- Update stale /Rotate gap and deskew doc
- Add production documentation (README, getting-started, API ref, CLI ref)
- Add hardware compatibility section to all three docs
- Add planned platform support roadmap to hardware compatibility sections

### Encode

- Review pass — rename PngEncoder, improve docs and CMYK notes
- Hardening pass — API consistency, overflow guards, exhaustive matches

### Font

- Implement Type 3 paint-procedure fonts; hardening pass across pdf_interp

### GPU

- Research-driven overhaul — hardware backend, pinned memory, raw CUDA driver API
- Hardening pass — safety, correctness, and documentation
- GPU supersampled AA fill via 64-sample warp-ballot CUDA kernel
- Tile-parallel analytical fill kernel + ROADMAP update
- Fix three correctness bugs found in tile_fill review pass
- ICC CMYK→RGB kernel + pdf_interp wiring (matrix path)
- Review pass on ICC CMYK→RGB kernel and baking code
- Validate GPU AA fill parity against CPU reference
- Harden AA parity tests — shared GpuCtx, cleaner geometry, better failure messages
- AVX-512 CMYK→RGB for icc_cmyk_to_rgb_cpu clut=None path
- Harden AVX-512 CMYK path — remove dead code, fix test, clean idioms
- Calibrate dispatch thresholds; add threshold_bench binary
- Hardening pass on threshold calibration code
- Add nvJPEG2000 decoder for JPXDecode GPU fast path
- Hardening pass — sub-sampling guard, OOM cap, #[expect] casts, build.rs dedup
- Hardening pass 2 — status codes, SubSampledComponents, edition-2024 idioms
- Fix 'terminate called recursively' crash on malformed J2K
- Hardening pass — shim destroy fns, build env override, error surfacing
- Implement rotate_gpu via nppiRotate_8u_C1R_Ctx
- Extract shared CUDA driver init into gpu::cuda module
- Review fixes — shared DeviceBuf, context cleanup, checked arithmetic
- Fmt + clean all remaining clippy warnings
- Extract CMYK conversion, CPU compositing, and tile/fill helpers into submodules
- Hardening pass — fix HALTON3 table, NaN propagation, and dead code

### Other

- Hardening pass on all five modules
- Fix semicolon_if_nothing_returned and too_many_lines
- Hardening pass — fix transfer bug, deduplicate, rename, cleanup
- Hardening, logic, and correctness pass
- Hardening pass on golden.rs and generate.sh
- Eliminate per-span heap allocations in hot paths
- Hardening pass on allocation-fix commit
- Silence poppler stderr; route diagnostics to log crate
- Harden log callback — atomic fn ptr, correct safety docs, lossy UTF-8
- Add synthetic fill benchmark comparing vello_cpu vs pdf-raster
- Add PDF content stream tokenizer and operator decoder
- Hardening pass on tokenizer, operator decoder, and lib
- Wire font pipeline for text showing (Tj TJ ' ")
- Hardening pass — correctness, security, and idiom fixes
- Implement image XObject rendering (CCITTFaxDecode / FlateDecode)
- Hardening pass on image XObject pipeline
- Hardening pass — logic, duplication, and idioms
- Eliminate duplication, extract shared helpers, split god files
- Fix image SMask — skip images whose soft mask can't be decoded
- Harden SMask decoding — validation, overflow safety, correct defaults
- Implement Form XObject recursive execution
- Implement Encoding/Differences array for Type 1 / TrueType fonts
- Implement ExtGState opacity and graphics parameter overrides
- Refuse to open PDFs containing JavaScript
- Implement W/W* clip path operators
- Harden clip path implementation
- Split Phase 1 into blocking/nice-to-have/parking-lot, ordered by priority
- Add hardware context to Phase 2/4; corpus note on image decode dominance
- ICCBased/Indexed/CMYK colour spaces + fmt cleanup
- Hardening pass on colour space decoding
- Wire ExtGState blend modes (BM key)
- CCITTFaxDecode Group 3 1D (K=0) support
- Inline images (BI ID EI) + RunLengthDecode filter
- Shading (sh) — axial and radial gradients wired through
- Mark Phase 1 complete; update preamble to reflect native-only CLI
- Phase 2 complete; expand Phase 4 GPU strategy
- Add Phase 2.5 AVX-512 specialisation items
- Mark Phase 4 item 1 (nvJPEG) complete
- Mark Phase 4 nvJPEG complete, expand implementation notes
- Phase 2.5 — mark implemented CPU/SIMD features as complete
- API correctness pass against current stable docs
- Hardening pass + ROADMAP API audit notes
- Type 0 / CIDFont support + OOM safeguards
- Hardening + fmt pass on Type 0 / CIDFont code
- Implement PatternType 1 tiling patterns (scn/SCN)
- Implement text render modes 4–7 (text-as-clip)
- Exact inverse-affine image sampling for rotated/sheared images
- Implement Type 4/5 Gouraud mesh; fix OOM test
- Hardening pass on Type 4/5 mesh and BitReader
- Implement Types 6/7 (Coons patch / tensor-product patch mesh)
- Hardening pass on Types 6/7 implementation
- Implement JBIG2Decode via hayro-jbig2
- Implement Optional Content Group (OCG / layer) support
- Implement annotation appearance rendering (PDF §12.5)
- Mark Phase 1 parking lot complete
- Implement PDF transparency groups (§11.6.6)
- Defer halftone screens (print RIP only), Phase 3 effectively complete
- Wire GPU tile fill into fill_path dispatch + refactor
- Mark Phase 4 item 3 fully complete
- Review pass on GPU fill helpers
- Bake ICCBased CMYK CLUT from moxcms for accurate colour conversion
- Mark Phase 2.5 complete; Phase 4 ICC item complete
- Implement bpc 2, 4, 16 image decoding
- Harden bpc 2/4/16 image decoding
- Implement CCITTFaxDecode K>0 (Group 3 mixed 2D / T.4 MR)
- Implement Type 1 (function-based) shading via pre-sampled grid
- Hardening pass — BBox intersection, casts, mul_add, dedup
- Record Apr 2026 benchmark results and /Rotate gap
- Fix /Rotate and CropBox page geometry
- Phase 0 complete, Phase 5 deskew algorithm decided
- New library crate — raster_pdf API + deskew foundation
- Mark CLI thin-wrapper item complete
- Mark GPU deskew rotation complete
- Hardening, idiom, and logic review pass
- Close Phase 5, open Phase 6
- Implement UserUnit scaling and effective_dpi
- Add RenderDiagnostics (Phase 6)
- Mark UserUnit and RenderDiagnostics complete in Phase 6
- Harden RenderDiagnostics — fix rotated-page PPI bug, add const guard
- Add suggested_dpi (Phase 6 DPI auto-selection hint)
- Add render_channel — pipelined render + OCR (Phase 6)
- Harden render_channel and suggested_dpi
- Consolidate obj_to_f64, migrate tiling.rs to shared helpers
- Consolidate 4 resolve_dict variants into mod.rs
- Hardening pass on obj_to_f64/read_f64_n/tiling
- Hardening pass on resolve_dict/resolve_stream_dict
- Fix tokeniser hang on `>>`, fix render_page example, clean unused import
- Hardening pass — fix sentinel loop, EOF backslash, cast cleanup, new tests
- Cow/Box audit — eliminate redundant allocations
- Hardening pass — logic fixes, edge cases, dead code, new tests
- Fix tokeniser hang on stray ) and ] outside container contexts

### Raster

- Add image module (Phase 2 step 5)
- Harden image module (review pass)
- Add shading module — axial, radial, function patterns + Gouraud triangles
- Hardening pass on shading and transparency modules
- Hardening pass on fill, stroke, and glyph modules
- Tier 2 — [u16;16] compositing fast path with LLVM auto-vec
- Tier 3 — two-pass counting-sort eliminates per-row Vec allocs
- Sparse nonempty_rows fill loop — skip empty scanlines
- Hardening pass — overflow guards and invariant fixes
- Eliminate draw_span / draw_span_band duplication via RowSink
- Hardening pass on RowSink commit
- Aa_coverage_span with AVX-512 BITALG acceleration
- Hardening pass on aa_coverage_span
- Movdir64b non-temporal solid fill (Phase 2.5 item 3)
- Hardening pass on movdir64b fill
- Eliminate duplicate simple_pipe/make_clip/make_pipe test helpers
- Hardening pass — simple_pipe delegates to make_pipe, add docs

### Refactor

- Final dedup pass — unused imports, missing pub(super) docs, regenerate golden refs

### Renderer

- Extract GPU fill paths and annotation rendering into submodules
- Hardening pass — fix shear transform, silent failures, depth limit

### Resources

- Extract function eval and patch machinery into submodules
- Extract codec decoders and inline parser into submodules
- Hardening pass on function eval and patch machinery
- Hardening pass — fix overflow, polarity, and silent truncation bugs

### Testing

- Shellcheck clean, add dry-run to both scripts
- Feature-matrix benchmark + build helper, gitignore bins
- Hardening pass on all four shell scripts
- Golden image regression suite


