# Changelog

All notable changes to this project will be documented in this file.

## [1.0.1] - 2026-05-13

### Bug Fixes

- Add version = "1.0" to optional gpu path deps in rasterrocket-interp and rasterrocket
- Resolve clippy warnings in args.rs (match wildcard, clone_from)
- Remove version from optional gpu dep (publish = false; not on crates.io)
- Remove self-referential dev-deps that block crates.io upload
- Update gpu:: → rasterrocket_gpu:: in bin and integration tests

### Chores

- Rename repo to rasterrocket, enable crates.io publish
- Rename to rasterrocket-color, add crates.io metadata
- Rename to rasterrocket-render, add crates.io metadata
- Rename to rasterrocket-encode, add crates.io metadata
- Rename to rasterrocket-font, add crates.io metadata
- Rename to rasterrocket-parser, add crates.io metadata
- Rename to rasterrocket-interp, update dep aliases
- Rename to rasterrocket, update dep aliases
- Rename to rasterrocket-cli, binary name rasterrocket
- Mark gpu, pdf_bridge, bench as publish = false
- Rename pdf-raster → rasterrocket in source comments and string literals
- Sweep all stale pdf_raster / pdf_interp / pdf-raster name references
- Rename CLI binary from rasterrocket to rrocket
- Rename gpu crate to rasterrocket-gpu; fix CI package names; add GPU feature stubs
- Update Cargo.lock after removing self-referential dev-deps
- Mark internal crates publish = false; yank from crates.io

### Documentation

- Update ARCHITECTURE.md for v1.0.0 — pdf crate, libdeflate, session API, module paths
- Rename pdf-raster → rasterrocket throughout all documentation
- Rename pdf-raster → rasterrocket
- Clarify two-crate public surface; update README crate map and descriptions

## [1.0.0] - 2026-05-13

### Bug Fixes

- Apply tint to PaintType 2 tiling patterns
- Hoist samples.len() i64 conversion in test helper
- Make Phase 1+2 `n` per-region, not per-walk
- Phase 4 binding-slot mismatch + cross-stage barrier
- Hardening pass on B5/B6/B7 — qtable indexing, subsampling gate, loud errors
- Second hardening pass — CUDA kernel stub, quant-selector gate, overflow, test accuracy
- Multi-pass review — threshold, grayscale cs, failed-init sentinel, kernel guards, wire-value pin, luma-table selection, bench fixes
- Snapshot block_in_mcu/z_in_block post-rotation, not pre-rotation
- Guard and widen block_idx arithmetic against u32 overflow
- Require nxt decoded at least one symbol before accepting sync
- Prevent signed i32 overflow in idct_1d fixed-point multiplications
- Vulkan implies gpu-jpeg-huffman in pdf_interp + pdf_raster
- Avoid SIGPIPE under pipefail in Vulkan ICD detection
- Restore missing #[test] on raster_options_with_pages_none_is_valid
- Make deskew::detect and deskew::rotate pub(crate) to stop leaking raster::Bitmap
- Gate BackendPolicy::ForceVaapi on #[cfg(feature="vaapi")]
- Mark SessionConfig #[non_exhaustive] to handle feature-gated prefetch field
- Suppress unnecessary_wraps on rotate_inplace, pub(crate) → pub on rotate_cpu in private module

### Chores

- Commit bench data, wiki pages, ignore __pycache__
- Release v1.0.0

### Documentation

- Replace stale Phase 1/2 refs in build_adjusts
- Cross-reference §8.9.6 in blit_image Mask arms
- Refresh module + variant docs post-ICC/Separation
- Pin TransferSet.device_n shape rationale
- Reframe scale-kernel deferral as Phase 10 task 4
- Note Phase 8 Phase A shipped as OSS artifact
- Update Vulkan docs — parallel-Huffman JPEG wired in, not CPU-fallback
- Fix ocr_from_frame footgun in README snippet, add OCR wiki links
- Replace ocr_from_frame footgun with wiki link in crate doc
- Expand release_gpu_decoders doc with pool.broadcast example, no-op note, and Panics
- Update api-reference.md for all pre-1.0 API cleanup changes
- Move benchmarks to GitHub wiki
- Update README for v1.0.0 — version tag, What's new, bench table
- Update getting-started for v1.0.0 — tags, RasterOptions, suggested_dpi fix
- Fix api-reference — move suggested_dpi to PageDiagnostics, remove stale RenderedPage method
- Update Benchmarks wiki — v1.0.0 tool version, v0.9.1 five-mode results, extended regression history
- Add v1.0.0 release section to ROADMAP
- Fix suggested_dpi prose in getting-started; add parallel-Huffman post-mortem wiki page
- Prep v1.0.0 release notes

### Features

- Decode Type 0 sample streams, clip to Range
- Add ColorSpace taxonomy for gstate tracking
- Add fill/stroke_color_space slots
- Plumb cs/CS operators into gstate color_space slots
- ColorSpace::convert_to_rgb tint → sRGB
- Route uncoloured-pattern tints through ColorSpace
- Store lookup_id on Indexed variant
- Wire ICC CLUT through IccBased convert_to_rgb
- Wire eval_function into Separation tint
- Scaffold module + error taxonomy behind gpu-jpeg-huffman
- 32-bit BE bitstream packer
- 2-tier quick+full codetable builder
- Scalar CPU reference decoder
- Implement upload_async + download_async on the trait
- ScanParams + record_scan trait method (API-only)
- Blelloch exclusive scan kernel + CUDA dispatch
- Record_scan + cross-backend bit-identity tests
- Phase 1 intra-sequence-sync kernel + CUDA dispatcher
- Record_huffman + cross-backend Phase 1 parity tests
- Phase 2 inter-sync kernel + CUDA + Vulkan dispatchers
- Phase 3 wires Blelloch scan over symbol counts
- Phase 4 re-decode + write; end-to-end on CUDA + Vulkan
- Phase 4 typed decode_status surface
- JpegPreparedInput wraps CPU pre-pass for GPU dispatch
- Retain DC codebooks for the on-GPU bitstream walker
- CPU oracle for the JPEG-framed symbol stream
- JPEG-framed HuffmanPhase variants + params surface
- Slang jpeg_phase1_intra_sync with real JPEG framing
- CUDA jpeg_phase1_intra_sync mirror + expanded Phase4FailureKind
- B2d — JPEG Phase 1 dispatch + oracle parity on real JPEG
- B2e — JpegPhase2InterSync kernel + host dispatcher + parity tests
- B2f — JpegPhase4Redecode kernel + full 4-phase JPEG dispatcher + parity tests
- B3 — quality-aware subsequence size from qtable magnitude
- B4 — idct_dequant_colour Slang kernel (IDCT + dequant + YCbCr→RGB)
- B5/B6 — JpegGpuDecoder + IdctParams + IDCT kernel wiring
- Wire parallel-Huffman JPEG decoder into pdf_interp + pdf_raster
- Gpu-jpeg-huffman gate harness + aggregator
- Generic JpegGpuInit<T> + Vulkan TLS slot + ensure_jpeg_vk_huffman
- Add jpeg_vk field + set/take_jpeg_vk to PageRenderer
- Route lend/reclaim_decoders to Vulkan JPEG decoder when active
- V0.9.0 four-mode bench harness — CPU / nvJPEG / CUDA-Huffman / Vulkan-Huffman
- V0.9.1 five-mode bench harness + PDF_RASTER_HUFFMAN_THRESHOLD env override
- Impl Default for RasterOptions (dpi=300, all pages, no deskew)
- Add prescan_session; hide doc()/resolve_page()/prescan_page to stop leaking pdf::Document
- Add pub mod session grouping open_session/render_page_rgb/prescan_session
- Wire GPU Phases 1-4 parallel-Huffman into production decode path

### Other

- Remove unsafe get_unchecked from blit_image hot loop
- Pin PDF §8.9.6 mask-no-smask invariant in blit_image
- Tighten eval_sampled smell pass
- Cap tiling-pattern recursion depth
- Extract check_image_bytes_len for unit testability
- Clippy --fix sweep on test modules
- Kill remaining structural clippy lints on --tests
- Tighten clippy expects + AA_GAMMA test derivation
- Tighten pattern-tint test smells
- Wire DeviceN N=1 through tint fn
- Unify ICC-test tolerance + add blue + neutrality
- Tighten fast-path corpus + idioms
- Pre-existing clippy + idiom sweep
- Rename JpegGpuError::BackendError to ::Dispatch
- Tighten contract + 5 new edge-case tests
- Drop unused _is_ac param + tighten loop invariant
- Drop cargo-cult wrapping_sub + document caller contract
- Test-only module + 4 peek16 boundary tests
- Scan dedup + compile-time kernel-slot drift guard
- Simplify pass on A7 — dedup, derive, fail-modes, dead-code
- A7 hardening pass — overflow guard, fixture dedup, dead code
- A7b hardening pass — lift PHASE1_THREADS to backend::params
- A8 hardening pass — drop redundant guard + dead binding
- A9 hardening — dedup Phase 2 outcome + retry bound
- A11 hardening — dedup scan oracle + pin Phase 1+2 first
- A12 hardening — bounds-check writes + dedup decode + dedup scan
- RAII guard — accurate doc + #[must_use]
- Visitor pre-check + assert; lift validate_canonical_table
- A13 — defensive max_iters; drop unused Phase 4 arg; idiomatic vec_04
- Typed error mapping + SOS/SOF guard + drop ScanClass
- ZRL overflow, AC size cap, error context, dedup mcu_count
- Hard asserts on shift invariants + overflow-safe refill
- Dedup HuffmanParams validate helpers; clarify per-field docs
- JPEG kernel — cap DC category and AC size; pipeline structural tests
- Pin Phase4FailureKind discriminants at build time
- B2d hardening pass
- Baseline A/D/H results — Vulkan skipped (no ICD on bench machine)
- Full five-mode results — CUDA Huffman beats nvJPEG on 7/10, Vulkan Huffman flat vs CPU JPEG

### Refactor

- Collapse per-arg let-_ on kernel launch builders
- Use TransferLut::IDENTITY in test LUTs
- Share empty_doc helper across test modules
- Add TransferLut::INVERTED const, dedup test LUTs
- Thread Document through convert_to_rgb
- RAII guard for dispatcher device buffers
- Extract visit_canonical_codes; collapse 3 call sites
- Lift BitReader into shared `jpeg::bitreader` module
- Genericise decode_dct over B: GpuBackend

### Testing

- Unblock 4 inline-image decode tests
- Restore stitching wrong-bounds fallback test
- Use static INV instead of Box::leak in transfer test
- Make stitching fallback test actually pin fns[0]
- Add make_doc_with_stream helper for ICC/stream tests
- ICC sRGB-profile happy-path round-trip
- Pin fast-path vs general-path within 1 LSB
- Synthetic Huffman encoder oracle
- Phase 1 CPU oracle — validates kernel spec
- Phase 2 inter-sync CPU oracle
- A13 corpus + Phase 1 boundary-snapshot semantics
- B5 — DRI ∈ {1,8,64} correctness tests
- B7 — IEEE 1180 pixel-diff parity tests vs zune-jpeg
- Add YCbCr 4:4:4 parity tests for blocks_per_mcu=3

## [0.9.2] - 2026-05-11

### Bug Fixes

- U32 overflow in three scale_mask kernels + golden tests
- 14 pre-existing clippy lints under --features ...,vaapi --tests
- Clean clippy --tests (40 → 0 warnings)
- Use PDF font dict Widths for simple fonts (PDF §9.2.4)

### Chores

- Release v0.9.2

### Documentation

- Document alloc_device_zeroed compute-queue contention
- Prep v0.9.2 release notes

### Features

- GpuBackend::record_zero_buffer on CUDA + Vulkan

### Other

- PageCursor over PageSet, fix sparse iteration hang
- Simplify PageCursor on top of 041465a
- Drop dead public helpers + downgrade clip255 to private
- NaN regression-pins + drop clip255 + 2x #[expect] removed
- Collapse cmyk_to_rgb to chained u8::saturating_sub
- Unreachable! over debug_assert, cs_to_rgb invariants
- Kill per-call Vec alloc in SVE2 + rename popcnt → aa_coverage
- State-first validation + shared fill_size helper
- Saturate PDF width casts via pdf_width_to_i32

### Refactor

- Route ppm + shading helpers through color::convert
- Delete dead popcnt_aa_row + all 5 tier impls
- Unify 8 scale kernels behind one ImageSource path

### Testing

- Pin both branches of is_identity_rgb gate
- Structural rotate_gpu pin + clippy clean under gpu-deskew
- CPU rotation invariants kill 17 cargo-mutants survivors

## [0.9.1] - 2026-05-11

### Chores

- Release v0.9.1

### Documentation

- Prep v0.9.1 release notes

### Other

- Backend trait + Vulkan hardening + cleanups (#1)
- Drop dead C++-era code, tighten visibility, kill -1.0 sentinel
- Hardening + simplify review on top of 3c2804c
- Delete dead RenderedPage::suggested_dpi delegate
- Use {f64,u16}::midpoint in shading/patch.rs tests
- Delete dead StateFlags accessors and GraphicsState delegates
- Delete more dead methods surfaced by mutants partial
- Delete dead PrefetchHandle API surfaced by mutants
- Finish the PrefetchStats deletion 0bb5f87 deferred
- Extract bounds-check helpers, fix stale doc ref
- Collapse single-bit StateFlags wrapper to plain bool
- Use u8::midpoint directly + From for u8→i16
- Simplify-pass fixes on 0711af1 scope
- Fail-loud-and-gracefully on all open/parse paths
- Reframe mutant-driven test names to describe behaviour

## [0.9.0] - 2026-05-09

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
- Scope dead_code expect to non-test builds in madvise.rs
- Preserve flate2's partial-decompression tolerance under libdeflate

### Chores

- Drop stale CUDA-12.8 path pins after driver/toolkit bump
- Audit init paths — already lazy, no change needed
- Add PGO + BOLT release build script
- Release v0.9.0

### Documentation

- Backfill v0.7.0 release notes
- Document why DevicePageBuffer / DeviceImageCache stay un-generified
- Mark Phase 10 task 1 partial — trait + CudaBackend shipped
- Phase 10 Task 3 follow-ups + scrub stale comments
- Phase 10 Task 4 — renderer migration + bench gate
- Prep v0.8.0 release notes
- Rewrite Phase 11 around the million-page-archive contest
- Drop internal-doc paths from public ROADMAP
- Phase 11 v0.9.0 release notes + contest results
- Full E1-E4 vs mutool, post-mortem, mutool defaults only
- Prep v0.9.0 release notes

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
- Add Document::get_page (O(log)) and page_count_fast (O(1))
- /Linearized detection + LinearizationHints API
- Posix_fadvise hints on Document mmap
- Contest_v11 binary skeleton + qpdf-based archive builder
- Contest_v11 four-event runners + competitor wrappers
- Apples-to-apples contest harness — disk write, mutool flags, dedup-defeating archive builder
- Vulkan-default Auto, env-var override, process-static GPU init

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
- Merge phase 10 task 1 — GpuBackend trait + CudaBackend skeleton

Brings in the in-progress Phase 10 work from the phase10-vulkan-backend
worktree:

- GpuBackend trait + *Params structs with state-machine + invariant docs
- CudaBackend: init, alloc, budget, record_* + submit_page / wait_page
- 5 existing kernels extracted into lib_kernels::{composite, soft_mask,
  aa, tile, icc} submodules; blit re-exported for symmetry
- BackendError::msg constructor; cuda_backend_smoke test
- gpu/cache mod.rs split into budget / eviction / promotion submodules
- Docstring on DevicePageBuffer / DeviceImageCache explaining why they
  stay un-generified until the trait grows an upload/download surface
  (task 1.10 blocked by 3.6 — see project memory)

No renderer migration yet; pdf_interp::renderer::page::gpu_ops still
talks to cudarc directly. Task 2 (Slang ports) and task 3 (Vulkan
backend) not started.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
- Phase 10 task 4 step 4 — bench gate, all measurable criteria PASS
- Collapse path-noise import + add u32::MAX edge-case test
- Drop dead linearization stub, idiomatic try_from, 3-level test
- Simplify-skill cleanup pass on page-tree descent
- Consolidate page-resolve paths, drop dead shading wrapper
- Drop redundant page-id cache, plumb single resolution
- Linearization cleanup + Object::{as_u32,as_u64} helpers
- Tighten as_u32/as_u64, drop xref obj_to_u32, lin fast path
- Minor tidy on madvise.rs (rustix import, test helper)
- Fix temp_file_with doc-comment
- Fix zlib decompression bomb in flate2 paths
- Libdeflater error uses Display, not Debug
- Fix E4 off-by-one, polite missing-arg UX, narrow pdftoppm cleanup
- Drop redundant install probes, tighten cleanup, fix target_bytes UX
- GPU-features re-bench across CUDA-cache, CUDA-nocache, Vulkan
- Plumb resolved policy, gate ForceCuda-without-features, tighten warnings

### Performance

- Libdeflate backend for FlateDecode (1.4-2.4x speedup)

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
- Lazy page resolution via O(log) descent

### Testing

- Scaffolding for logarithmic page-tree descent

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
- Simplify(gpu/jpeg): apply review findings — Option-typed quant tables,
shared MCU formula, marker constants, fail-fast on progressive JPEG

Three-agent simplifier review surfaced cleanups across the Phase 0 module
and the VA-API adapter.  This commit applies the findings the reviewers
agreed were clear wins; perf-only suggestions (codebook pooling, fused
peek+lookup+consume, #[inline(always)] on the bit reader) are deferred
until the bench gate is actually measured.

## Cleanups

- **Option-typed quant tables.**  Replace
  `quant_tables: [JpegQuantTable; 4]` + parallel `quant_present: [bool; 4]`
  with `quant_tables: [Option<JpegQuantTable>; 4]`.  Drops the
  presence-mask leak; `Option` carries the same one-bit signal in the
  type system.  Same shape applied in `CpuPrepassOutput`.  VA-API
  adapter still flattens to its own `[u8; 64] + [bool; 4]` shape (which
  is what the FFI buffers actually want).

- **Shared MCU formula.**  The MCU-count arithmetic appeared in
  `JpegHeaders::num_mcus` and was about to be duplicated again in
  `CpuPrepassOutput`.  Extracted to free function
  `headers::mcu_count(width, height, frame_components) -> u32`; both
  call sites delegate.  Replaces the redundant `num_mcus: u32` cache
  field on `CpuPrepassOutput` with a method.

- **`Display` for `DhtClass`.**  Dropped three triplicated
  `match class { Dc => "DC", Ac => "AC" }` blocks (in `dc_chain.rs`,
  `headers.rs`, `prepass.rs`) in favour of `impl Display for DhtClass`
  + a `pub const fn name(self) -> &'static str` accessor.

- **Marker-byte constants.**  `0xC0` (SOF0), `0xC4` (DHT), `0xDA` (SOS),
  `0xDB` (DQT), `0xDD` (DRI) appeared ~25 times across `headers.rs`.
  Promoted to module-level `const MARKER_*` so match arms and error
  variants name their markers per JPEG spec (Annex B) rather than
  requiring a hex-to-mnemonic mental lookup.

- **Fail-fast on progressive JPEG.**  The shared parser's contract was
  documented as "non-baseline streams are detected via `jpeg_sof_type`
  before calling parse" but nothing enforced it: callers got
  `MissingSof0` or `Truncated`, both unhelpful for routing.  The parser
  now calls `jpeg_sof_type` itself at the front door and returns a new
  `JpegHeaderError::NotBaseline` for `JpegVariant::Progressive`, which
  is unambiguous and actionable.  `JpegVariant::Other` (covering both
  unsupported SOFs and truncated-before-SOF) is intentionally not
  rejected here — those fall through to the existing
  truncation/missing-SOF0 paths.

- **Flatten the RST-boundary loop.**  `while let Some(&&rst) = peek()
  { if cond { ... } else { break } }` becomes
  `while peek().is_some_and(|&&r| cond) { let rst = next().expect(...); ... }`.
  One nesting level less; the exit condition is in the `while`.

- **Comment hygiene.**  Stripped commit-message-flavoured text from
  module headers (mod.rs's "in a later commit", canonical.rs's gpuhd /
  GPUJPEG provenance, prepass.rs's Phase-1 narration).  Kept WHY
  comments documenting non-obvious invariants (32-bit-host trap in
  the MAX_SAFE_MCUS check, RST byte-alignment, EXTEND wire format).

- **Fix "256 KB" → "128 KB" docstring drift.**  The codebook table is
  65 536 × 2 bytes = 128 KB, not 256 KB.  Comment was wrong since the
  table-format change went in.

## Behaviour-preserving (for callers)

- `CpuPrepassOutput::num_mcus()` is now a method (was a field).
- `CpuPrepassOutput::quant_tables[i]` is now `Option<JpegQuantTable>`
  (was `JpegQuantTable` with a parallel `quant_present[i]: bool`).
- `JpegHeaders::quant_tables[i]` similarly.
- New `JpegHeaderError::NotBaseline` variant; progressive JPEG now
  returns it instead of `MissingSof0`.

The VA-API adapter compensates internally — it still exposes the same
`(quant_tables: [[u8;64];4], quant_present: [bool;4])` shape to the
VA-API FFI buffers because that's what the parameter buffers want.

## Test status

- 84 jpeg::* tests pass (was 84; one progressive test re-pinned
  against the new `NotBaseline` error variant).
- 101 lib tests pass with `--features vaapi` (was 100; the VA-API
  adapter test was rewritten to verify the routable error message
  rather than the negation pattern that no longer applies).
- `cargo clippy -p gpu --no-deps`: 0 warnings.
- `cargo clippy -p gpu --features vaapi --no-deps`: 0 warnings.
- `cargo clippy --workspace --exclude pdf_bridge --no-deps`: 0 warnings.
- `cargo fmt --check` clean.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
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
- Merge feat/vaapi-decode-queue: VA-API JPEG decode queue

Routes all VA-API JPEG decodes through a single worker thread to eliminate
Mesa VCN driver mutex contention across Rayon workers.  See the feature
commit for full details.
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
- Cli, encode: implement --gray and --mono output flags

--gray: converts the rendered RGB bitmap to grayscale (BT.709 integer
coefficients: 2126·R + 7152·G + 722·B) and writes:
  - .pgm (P5 Netpbm) for PPM output mode
  - .png (gray color type) for PNG output mode

--mono: additionally thresholds the gray bitmap at 128 (< 128 → black,
≥ 128 → white, matching pdftoppm convention) and writes:
  - .pbm (P4 Netpbm, 1-bit MSB-packed) for PPM output mode
  - .png (gray PNG, values 0/255) for PNG output mode

New encode::write_pbm (P4 encoder) added with 6 unit tests covering:
header format, all-white/all-black rows, alternating checkerboard,
row padding for non-multiple-of-8 widths, and unsupported-mode rejection.

OutputFormat::extension() replaced by extension_with_mode(gray, mono) so
the file extension correctly reflects the actual output content; the startup
warning is removed.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
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

- Color, pdf_interp/image, gpu: extract cmyk_to_rgb_reflectance to color::convert

Move the reflectance formula R=(255−C)*(255−K)/255 from image.rs and
gpu::cmyk_to_rgb_pixel_scalar into color::convert::cmyk_to_rgb_reflectance.
Both callers used identical arithmetic; gpu already depends on color.
Other cmyk_to_rgb variants use different formulas and are left unchanged
with documented distinctions.  Tests for the reflectance formula migrate to
color::convert; icc_clut.cu comment updated.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Color, pdf_interp/renderer: promote gray_to_u8/rgb_to_bytes/cmyk_to_rgb_bytes to color::convert

Move three f64→u8 normalised-value converters out of renderer/color.rs into
color::convert where all colour arithmetic lives.  renderer/color.rs becomes
a thin adapter: re-exports from color, exposes RasterColor.  Tests for pure
conversion functions migrate to color::convert; RasterColor struct tests stay.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
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

- Gpu/nvjpeg + pdf_interp: GPU-accelerated JPEG decoding via nvJPEG (Phase 4 item 1)

gpu crate:
- New module gpu::nvjpeg (feature-gated: `gpu/nvjpeg`).
- Minimal raw FFI surface: nvjpegCreateSimple, nvjpegDestroy,
  nvjpegJpegStateCreate/Destroy, nvjpegGetImageInfo, nvjpegDecode — only
  what we use; no bindgen dependency.
- NvJpeg: reusable context (handle + state); decode() enqueues async GPU
  work on a caller-supplied CUstream; supports 1-component (Y) and
  3-component (RGBI) JPEG; rejects CMYK (4-component) cleanly.
- NvJpegDecoder: safe public wrapper that owns the CUstream pointer;
  decode_sync() does decode + cuStreamSynchronize so callers in
  pdf_interp (unsafe_code = "deny") get fully-synchronous semantics with
  no unsafe blocks on their side.
- build.rs: emit rustc-link-lib=dylib=nvjpeg and rustc-link-search for
  CUDA 12 install directory, only when CARGO_FEATURE_NVJPEG is set.
- Tests: new_does_not_panic, decode_gray_1x1, decode_empty_returns_error —
  all skip gracefully on machines without a GPU.

pdf_interp crate:
- Feature `nvjpeg` pulls in gpu/nvjpeg.
- decode_dct: new #[cfg(feature="nvjpeg")] fast path that calls
  decode_dct_gpu before zune-jpeg; falls back to CPU on any GPU error.
- decode_dct_gpu: calls NvJpegDecoder::decode_sync (fully safe), converts
  DecodedJpeg → ImageDescriptor, logs dimension mismatches.
- GPU_JPEG_THRESHOLD_PX = 262 144 (512×512): below this the PCIe transfer
  overhead exceeds zune-jpeg's decode time; above it nvJPEG at ~10 GB/s
  wins by 10-20×.
- Inline images always use the CPU path (typically small, not worth GPU
  dispatch overhead).
- resolve_image / PageResources::image / decode_inline_image: all updated
  with #[cfg(feature="nvjpeg")] NvJpegDecoder parameter using
  cfg-argument syntax to remain zero-overhead when the feature is off.
- PageRenderer: nvjpeg: Option<NvJpegDecoder> field + set_nvjpeg() setter
  so the CLI can attach a GPU decoder at startup.
- No unsafe code added to pdf_interp (unsafe_code = "deny" maintained).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Gpu/nvjpeg + pdf_interp: hardening pass on nvJPEG commit

#[allow] → #[expect] with reason strings on all 5 FFI type aliases.
Width/height casts replaced with try_from().expect() to catch sign-loss
at runtime. Null stream guard added to NvJpegDecoder::new. Updated stale
doc on NvJpeg::new (Returns None → Returns Err). Renamed subsampling →
_subsampling with corrected comment. Extracted try_cuda_stream() test
helper to deduplicate CUDA init across two unit tests. Removed redundant
rerun-if-env-changed line in build.rs; added warning when no CUDA lib dir
is found. decode_dct_gpu now distinguishes UnsupportedComponents (expected
CMYK path, debug) from other CUDA failures (warn). Doc comment added to
NvJpegDecoder re-export in resources/mod.rs.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
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

- Phase 1: color + raster foundation crates

Pure Rust foundation for the Splash rasterizer port. No rendering yet —
only the types, math primitives, pixel buffers, path geometry, edge tables,
clip, halftone screen, and graphics state that every later phase builds on.

Crate `color`: shared arithmetic (div255, lerp_u8, over_u8, premul/unpremul,
cmyk_to_rgb, byte_to_col, splash_floor/ceil/round), all pixel types (Rgb8,
Rgba8, Gray8, Cmyk8, DeviceN8) via bytemuck::Pod, and TransferLut.

Crate `raster`: Bitmap<P> with bytemuck row access and 1-bit AaBuf; Path /
PathBuilder state machine; De Casteljau curve flattening (MAX_CURVE_SPLITS=1024);
stroke-adjust snapping; XPath flat edge table with y0≤y1 invariant and aa_scale;
XPathScanner flat SoA intersection table (no per-row Vec); ScanIterator span
coalescing; Clip with Arc-shared path scanners; HalftoneScreen (Bayer/clustered/
stochastic); GraphicsState Vec-based save/restore stack.

All 69 unit tests pass; zero clippy warnings (-D warnings).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Modernise to edition 2024; fix all pedantic clippy warnings

- Switch both crates to edition = "2024"
- Replace all `#[inline(always)]` with `#[inline]` (pedantic lint)
- Replace lossy `as` casts with `u32::from()`, `f32::from()`, `u8::try_from()` etc.
- Add `#[must_use]` to all pure functions and constructors
- Add `# Errors` / `# Panics` doc sections where required
- Fix `many_single_char_names`: rename r/g/b/k locals to red/green/blue/black etc.
- Use `f64::midpoint` / `i32::midpoint` for De Casteljau subdivision
- Use `f64::mul_add` for fused multiply-add in pixel and geometry math
- Group `GraphicsState`'s many bool fields into a `StateFlags` bitflags struct
- Make pure query methods `const fn` throughout
- Replace `unsafe` f64→i32 casts in splash_floor/ceil with `to_int_unchecked`
  (with documented safety invariant) and i32::try_from for saturation
- Zero clippy warnings under -W clippy::all -W clippy::pedantic -W clippy::nursery
- All 69 tests pass; no `#[allow]` / `#[expect]` suppressions anywhere

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Hardening pass: zero pedantic clippy warnings, no unsafe, #[expect] for bounded float casts

- color::pixel: replace unsafe to_int_unchecked with const fn f32_to_u8 using
  #[expect(cast_possible_truncation)] — known false-positive for clamped values
- color::pixel: extract f32_to_u8 and rgb_to_cmyk_channel helpers, eliminating
  12+ duplicated call-site expressions; compile-time size assertions for all types
- color::convert: replace all `as f64`/`as i32` lossless casts with From::from
  in tests; fix doc backtick; add edge-case tests
- color::transfer: replace usize→u8 casts in test LUTs with std::array::from_fn
  + u8::try_from; remove redundant iter patterns
- color::mode: exhaustive match for bytes_per_pixel, pixel_count_to_bytes, from_u8
- raster::screen: replace unsafe to_int_unchecked with #[expect] for bounded cast
- raster::path: remove redundant clone in append test
- raster::scanner::iter: replace needless collect with .next().is_none()
- raster::state: fix float_cmp in tests; u8::try_from for LUT construction
- raster::xpath: doc backtick fix, const fn empty(), float_cmp fixes in tests

All 104 tests pass. Zero clippy warnings (--all -W pedantic -W nursery).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Hardening pass: edge-case guards, bug fix, docs, DRY across all modules

- Fix double-count bug in ScanIterator::next() coalescing logic
- Add debug_assert! guards for division-by-zero, overflow, and bounds
  in screen.rs (bayer_index, build_clustered, torus_dist, build_SCD),
  xpath.rs (dxdy division, aa_scale non-finite), state.rs (zero dims),
  bitmap.rs (row bounds with descriptive messages)
- Extract shared helpers: clamp_min_one (screen), aa_coords (clip),
  count_crossings (scanner), u32_to_usize with compile-time assertion
- Replace all #[allow] with #[expect(reason=...)] throughout; zero lazy suppressions
- Add missing /// doc comments on all pub items (fields, variants, constants)
  across pixel.rs, bitmap.rs, clip.rs, scanner, xpath, state, types
- Add # Panics sections to all methods that can panic
- checked_mul/checked_add for AA coordinate arithmetic in clip.rs
- ScreenParams::validate() made const fn; StateFlags bit-position table added
- 30+ new tests covering edge cases, invariants, and the iterator bug fix

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Hardening pass 2: eliminate false unwrap_or sentinels, fix empty-range signal, dedup

- Replace every unwrap_or(u32::MAX / 255 / 0) after a debug_assert!(...is_ok())
  with .expect("reason") so invariant violations fail loudly in release too
- Fix render_aa_line empty-range: set *x0=0, *x1=-1 instead of *x0=0, *x1=0
  (zero-width range was ambiguous with a valid single-pixel span at x=0)
- Remove dead `let _aa = ...` computation in clip_aa_line (no side effects)
- Guard Path::append / PathBuilder::close with debug_assert! on length invariants
- Add debug_assert!(P::BYTES > 0) before chunks_exact_mut in Bitmap::clear
- Guard build_adjusts reserved params (adjust_lines / line_pos_i) with debug_assert!
- Refactor detect_rect to single-pass collection (no iterator clone / double-filter)
- Remove two unfulfilled #[expect(cast_sign_loss)] on rem_euclid-to-usize casts
  (clippy does not fire for this pattern; annotations were noise)

All 143 tests pass; cargo clippy -D warnings clean.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Phase 2 step 1: raster::pipe compositing pipeline

Implements the full SplashPipe replacement:
- pipe/blend.rs: all 16 PDF blend modes (separable + non-separable RGB/CMYK)
- pipe/simple.rs: fast path for opaque solid fills (a_input=255, Normal, no shape)
- pipe/aa.rs: AA path for shape-coverage compositing (Porter-Duff over with per-pixel shape)
- pipe/general.rs: full PDF §11.3 compositing (soft mask, non-isolated groups, knockout, overprint, blend modes)
- pipe/mod.rs: PipeSrc, Pattern trait, PipeState, render_span<P> dispatcher
- state.rs: TransferSet<'a> borrowed view of transfer LUTs; transfer_set() method
- types.rs: BlendMode enum (16 variants, derives Default=Normal)

Zero clippy warnings under -D warnings + pedantic + nursery. 147 tests passing.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Phase 2 step 2: raster::fill — filled-path rasterization

Implements fill/eo_fill entry points (replaces Splash::fillWithPattern):
- Non-AA path: XPath → XPathScanner → ScanIterator span walk → clip → render_span
- AA path: XPath aa_scale → scanner in 4× coords → render_aa_line → clip_aa_line
  → draw_aa_line with AA_GAMMA table (splashAAGamma=1.5 precomputed)
- Bitmap::row_and_alpha_mut added to avoid split-borrow conflict on pixel+alpha rows
- Hard-coded AA_GAMMA[17] matches C++ aaGamma[] for splashAASize=4
- 5 unit tests; full suite now 185 tests, zero clippy warnings

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Phase 2 step 3: raster::stroke — stroke rasterization

Implements the full stroke pipeline (replaces Splash::stroke and friends):
- flatten_path: curve→line flattening via path::flatten::flatten_curve
- make_dashed_path: dash pattern application along subpath segments
- make_stroke_path: full stroke expansion — butt/round/projecting caps,
  miter/round/bevel joins, stroke-adjust hint emission
- stroke_narrow: hairline rendering — walks XPath segments scanline by scanline
- stroke_wide: make_stroke_path + fill (non-zero winding)
- stroke: top-level dispatcher — flatten → dash → narrow vs wide selection
- StrokeParams: groups all stroke state (lineWidth, lineCap, lineJoin, etc.)
- PathBuilder::pts_len() added for index tracking in makeStrokePath

Zero clippy warnings. 190 tests passing.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Phase 2 step 4: raster::glyph — glyph bitmap blitting

Implements GlyphBitmap + blit_glyph/fill_glyph (replaces Splash::fillGlyph2):
- AA mode: per-byte coverage → run-batched render_span_aa calls
- Mono mode: MSB-first 1-bit packed → per-row run detection → render_span_simple
- Both modes: clip_all_inside fast path (no per-pixel test) or per-pixel Clip::test
- fill_glyph: clip bbox test + blit_glyph convenience wrapper
- 7 unit tests covering AA paint, zero coverage, mono set bits, clip exclusion

Zero clippy warnings. 197 tests passing.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Hardening pass on all five modules
- Fix semicolon_if_nothing_returned and too_many_lines
- Hardening pass — fix transfer bug, deduplicate, rename, cleanup
- Add raster::transparency — PDF transparency group compositing

Implements begin_group/paint_group/discard_group and extract_soft_mask,
replacing Splash::beginTransparencyGroup / endTransparencyGroup /
paintTransparencyGroup.  Also adds Copy+Clone to TransferSet and
PipeState (all fields are reference types), alpha_plane/alpha_plane_mut
to Bitmap, and declares the module in lib.rs.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Hardening, logic, and correctness pass
- Add encode crate — PPM, PGM, and PNG bitmap writers

Adds a new `encode` workspace crate with three output format modules:
- `ppm`: Netpbm P6 binary writer for Rgb8/Bgr8/Xbgr8/Cmyk8/DeviceN8;
  CMYK→RGB via naïve ink-density subtraction matching pdftoppm.
- `pgm`: Netpbm P5 binary writer for Gray8; excludes stride padding.
- `png`: PNG writer via the `png` crate; Rgb8, Gray8 (via Mono8 mode),
  and Rgba8 (via Xbgr8 mode); Rgb8 bitmaps with an alpha plane are
  automatically promoted to RGBA PNG.

EncodeError unifies Io, Png, and UnsupportedMode variants with
std::error::Error impl and From conversions.  13 tests cover
roundtrips, CMYK conversion, clamping, stride-padding exclusion,
and unsupported-mode guards.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Add font crate — FreeType bridge, glyph cache, outline decomposition

Ports SplashFTFont / SplashFTFontEngine to safe Rust:
- engine: FontEngine owns FT_Library, assigns monotonic FaceId, loads faces
  from bytes or file path via freetype-rs 0.38
- face: FontFace owns FT_Face per instance (not thread-shared), rasterizes
  glyphs via make_glyph(), decomposes outlines via glyph_path(), reports
  advance via glyph_advance()
- hinting: load_flags() encodes the same hinting-mode decision tree as the
  C++ SplashFTFont constructor
- outline: decompose_outline() walks FreeType contour iterators and converts
  conic Béziers to cubics via degree-elevation (matching glyphPathConicTo)
- cache: GlyphCache wraps DashMap<FaceId, LruCache<GlyphKey, Arc<GlyphBitmap>>>
  for concurrent, per-face glyph caching across threads
- t3_cache: Type3Cache is an 8-slot MRU cache for Type 3 per-instance glyphs
- key: FaceId + GlyphKey (with 2-bit sub-pixel x-fraction, matching
  splashFontFractionBits=2)
- bitmap: GlyphBitmap owns rasterized pixel data (AA or mono packed)
- raster/path: add PathBuilder::cur_pt() needed by conic decomposition

35 unit tests + 1 doctest, clippy -D warnings clean.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Add pdf_bridge and cli crates — Phase 2 complete

pdf_bridge: C++ shim wrapping poppler-cpp for safe Rust FFI; exposes
Document, Page, RenderedPage, and format/render-params types.

cli: pdf-raster binary mirroring all pdftoppm flags; renders pages in
parallel via rayon with PPM/PGM or PNG output.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Pdf_bridge + cli: hardening and correctness pass

pdf_bridge:
- build.rs: use dpkg-architecture for portable multiarch lib path;
  assert clearly when versioned .so is missing; collapse nested if
- poppler_shim.cc: null-guard set_data_dir; add explicit SHIM_FORMAT_RGB24
  case to format switch so the fallback is intentional, not silent
- lib.rs: propagate DataTooLarge error instead of silently clamping
  oversized from_bytes buffers; unify repeated i32→u32 cast into
  nonneg_i32_to_u32 helper; add pts_to_pixels helper with edge-case
  handling (zero/negative pts, non-finite/zero dpi, overflow clamp);
  row() returns Option<&[u8]> instead of panicking; error source()
  chain wired for all variants; add 3 new unit tests

cli/render.rs: replace two identical stride-copy loops with generic
  rendered_to_bitmap<P, BPP>; report JPEG/TIFF as UnsupportedFormat
  error instead of unreachable!; wire error source() chain

cli/main.rs: warn (don't silently clamp) when requested page range
  exceeds document; empty page-set after filter is a hard error;
  sort errors by page number; print full error cause chain

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Phase 3: SIMD acceleration — blend, composite, popcnt, glyph unpack

raster/simd/blend.rs:
- blend_solid_rgb8: AVX2 path tiles 96-byte (LCM of 3 and 32) pattern,
  writes three 256-bit stores per 32-pixel chunk; scalar tail
- blend_solid_gray8: AVX2 set1_epi8 + 32-byte stores; scalar uses fill()
- Runtime dispatch via is_x86_feature_detected!("avx2")

raster/simd/composite.rs:
- composite_aa_rgb8: scalar Porter-Duff source-over with per-pixel shape;
  identical logic to pipe/aa.rs but callable from outside the pipe

raster/simd/popcnt.rs:
- popcnt_aa_row: three tiers — AVX-512 VPOPCNTDQ (64 bytes/iter),
  popcnt64 (8 bytes/iter), scalar; runtime dispatch

raster/simd/glyph_unpack.rs:
- unpack_mono_row: SSE4.1 path broadcasts each source byte into __m128i,
  ANDs with bit-isolation mask, cmpeq-with-zero gives 0xFF/0x00 per pixel;
  16 bytes output per iteration; scalar tail

pipe/simple.rs: dispatch to simd::blend_solid_{rgb8,gray8} for ncomps=3/1
  under cfg(all(target_arch = "x86_64", feature = "simd-avx2"))

glyph.rs: byte-aligned mono blit uses simd::unpack_mono_row; strided path
  retains scalar bit-extraction

Features: default = ["simd-avx2"], simd-avx512 = ["simd-avx2"]
Tests: 342 total (226 in raster, +41 new SIMD scalar+runtime tests)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Phase 4: tile-level rayon parallelism for fill (item 28)

raster/Cargo.toml: add `rayon = ["dep:rayon"]` optional feature.

bitmap.rs: add `BitmapBand<'bmp, P>` — a borrowed mutable view into a
  horizontal strip of a `Bitmap<P>`, with absolute-y row access.
  Add `Bitmap::bands_mut(n_bands)` which splits data/alpha planes into N
  disjoint `BitmapBand`s via `split_at_mut` (no unsafe, no copies).

fill.rs: under `#[cfg(feature = "rayon")]`:
- `PARALLEL_FILL_MIN_HEIGHT = 256` — threshold below which sequential fill
  is faster than rayon thread spawning
- `fill_parallel` / `eo_fill_parallel` — public entry points matching the
  existing API plus an `n_bands` argument
- Falls back to sequential for AA mode (shared AaBuf), single-band, or
  small fills below the height threshold
- Non-AA path: builds XPath + XPathScanner once (read-only, shared),
  splits bitmap into bands, dispatches via rayon par_iter
- 3 new tests: parallel matches sequential pixel-for-pixel (n_bands=4),
  single-band matches sequential, eo variant matches

Test counts: 226 (default) / 229 (--features rayon); 0 failures.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Phase 4: GPU acceleration — composite_rgba8 and apply_soft_mask (items 29-30)

New crate crates/gpu/:
- build.rs: compiles CUDA kernels to PTX via nvcc (-arch=sm_120, -O3,
  --use_fast_math); nvcc path discovered from env var or standard paths
- kernels/composite_rgba8.cu: Porter-Duff source-over for RGBA8 pixels;
  early-out for fully transparent/opaque source; correct alpha-blending
- kernels/apply_soft_mask.cu: per-pixel alpha × mask multiply saturate
- src/lib.rs: GpuCtx holding Arc<CudaStream> and compiled CudaFunction
  handles; threshold-gated dispatch (GPU only above 500k pixels); CPU
  fallbacks composite_rgba8_cpu / apply_soft_mask_cpu always available;
  GpuCtx::init() returns Err gracefully when no CUDA device is present
- 8 tests: 6 CPU-only (always run) + 2 GPU round-trip tests that compare
  GPU output to CPU fallback and skip if no CUDA device

Workspace: added crates/gpu to members.

Tests: 351 total (8 new in gpu, all passing including GPU tests on RTX 5070).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Harden SIMD and GPU: replace lazy allows, add precondition checks

- Replace all `#![allow(unsafe_code)]` blanket module allows in simd/
  with a single `#[expect]` on simd/mod.rs, annotated with reason
- Add `debug_assert!` precondition bounds checks to the two scalar
  blend functions that had none (rgb8 and gray8)
- Replace every `#[allow(...)]` in gpu/src/lib.rs with `#[expect(...)]`
  plus a why-reason so stale suppression is caught at compile time
- Replace `#[allow(unused_results)]` builder chains with explicit
  `let _ = builder.arg(...)` which is clearer and lint-clean without an allow

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- D65609f hardening: AVX2 preconditions, cfg idiom, run_shape alloc, composite doc

- blend.rs: add debug_assert! bounds checks at entry of both AVX2 fns;
  the SAFETY comments claimed "by construction" but nothing enforced it
- blend.rs: collapse two-line #[cfg] pairs in dispatch fns into
  #[cfg(all(target_arch, feature))] — consistent with the AVX2 fn defs above
- glyph.rs: hoist run_shape Vec above the row loop in blit_aa; it was
  re-allocated (Vec::new) on every row and immediately cleared, causing
  repeated heap allocations per glyph; now allocated once and cleared per row
- composite.rs: replace "same logic as pipe/aa.rs" comment with an explanation
  of why it is intentionally a standalone copy (different call signature,
  no PipeState dependency needed for transparency group callers)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- 4febab0 hardening: remove dead guard, drop unused param from fill_band

- fill_band: remove the dead `y < 0 || y >= band.y_start + band.height`
  guard inside the row loop — y is produced from
  scanner.y_min.max(y_band_min)..=scanner.y_max.min(y_band_max) where
  y_band_min >= 0 (band.y_start is u32), so the condition is always false
  and the #[expect(cast_sign_loss)] that went with it was hiding a dead branch
- fill_band: remove the _vector_antialias: bool parameter — the caller
  already falls back to sequential fill_impl for AA mode before calling
  fill_band, so the parameter was always false and never read; dropped the
  leading underscore suppress-trick and removed it from the call site too

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- 0816b13 hardening: fix unused_results, configurable CUDA arch, build.rs cleanup

lib.rs:
- builder.launch(cfg)? returns Result<Option<timing_events>, _>; the ?
  propagates errors but the Ok payload (timing events) was silently dropped,
  triggering the -D unused-results workspace lint. Fix: let _ = unsafe {
  builder.launch(cfg) }? to discard the timing events intentionally
- Remove two false #[expect(cast_possible_truncation)] on .min(255) as u8
  casts — clippy sees the bound and does not warn, so #[expect] was unfulfilled

build.rs:
- Remove dead #![allow(missing_docs)]: build.rs is a binary, missing_docs
  never fires there; the allow was suppressing nothing
- Hardcoded -arch=sm_120 (Blackwell-only) replaced with an env-configurable
  CUDA_ARCH variable defaulting to sm_80, which runs on Ampere/Ada/Hopper/
  Blackwell; add cargo:rerun-if-env-changed=CUDA_ARCH
- .unwrap() on path conversions replaced with .expect("...non-UTF-8") so
  failures name the actual problem
- nvcc name included in panic message so users know which binary failed

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- 85c8100 hardening: FaceId wrap guard, bottom-up bitmap guard, trim error source

engine.rs:
- alloc_id: was const fn with wrapping_add and a doc comment claiming FaceIds
  are never reused — wrapping would silently reuse them at 2^32 loads; add
  debug_assert!(next_id < u32::MAX) that fires before the wrap, make fn non-const
  (debug_assert is not const), and correct the comment to "wrap-around never
  occurs in practice" rather than "never reused"
- LoadError: remove the explicit source() { None } impl body — the default
  impl inherited from std::error::Error already returns None; freetype::Error
  does not implement std::error::Error so cannot be chained regardless

face.rs make_glyph:
- ft_bmp.pitch() is negative for bottom-up FreeType bitmaps; previously this
  was silently taken as unsigned_abs() and used as row stride, producing
  corrupted glyph data (row indexing goes backwards); now: if pitch < 0 return
  None (treated as a render failure, matching how other FT errors are handled)
- Add #[expect(cast_sign_loss)] on the now-guarded cast with reason

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Add comparison and benchmark test suite

tests/compare/compare.sh: renders the same pages with pdftoppm and
pdf-raster at a given DPI, diffs each page via ImageMagick RMSE, and
reports per-page pass/fail against a configurable threshold.

tests/bench/bench.sh: uses hyperfine to measure wall-clock throughput
of both binaries across the five fixture PDFs and multiple DPIs, then
emits a combined JSON result file with a pages/second summary table.

Fixtures (tests/fixtures/): five PDFs spanning 7–576 pages and 116KB–
50MB covering text-only, mixed, and image-heavy documents.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
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
- Fix font rendering: correct pixel size and charmap resolution

Three bugs prevented text from appearing:

1. FreeType pixel size never set — `FontFace::new` now calls
   `set_pixel_sizes(0, size_px)` so FreeType rasterizes at the
   correct scale instead of an arbitrary internal default.

2. Glyph charmap bypass — `resolve_gid` fell back to `char_code`
   directly as a GID, skipping FreeType's charmap.  Now falls through
   to `face.get_char_index()` for standard encodings (WinAnsi,
   MacRoman, Standard) where byte values in the printable ASCII range
   equal their Unicode codepoints.

3. Font loaded at wrong size — the cache passed `font_size` (the Tf
   operand, often 1.0 in PDF) to FreeType.  The real render size is
   `font_size × Tm[2×2] × CTM[2×2]` (the text rendering matrix).
   `FontCache::get_or_load` now takes `trm: [f64; 4]` (the full Trm
   2×2 submatrix) as the size descriptor, keyed by all four f64 bits.
   `show_text` computes Trm via the new `mat2x2_mul` helper in gstate.

Result: fixture-a.pdf page 1 now matches poppler at 2496/2498 dark pixels
(1.2% total pixel diff); fixture-b.pdf at 1.8% diff.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Hardening pass: font rendering correctness, safety, and clarity

face.rs:
- Guard NaN/Inf in size_f before u16 cast (was UB for NaN inputs)
- Change text_scale degenerate check from == 0.0 to < EPSILON
  to catch subnormal inputs; apply same fix in glyph_path
- resolve_gid: fallback on charmap miss returns GID 0 (.notdef)
  instead of char_code, which was wrong for Type1 fonts
- Add pitch < row_bytes guard to prevent slice-bounds panic on
  corrupt FreeType bitmaps (adversarial font data)
- Use checked_mul + ? for row*pitch to prevent overflow on
  pathologically large glyphs

font_cache.rs:
- Recover from poisoned mutex rather than panicking, so one bad
  thread doesn't kill all subsequent page renders
- Comment explaining why entry API cannot be used here
- Extract trm_pixel_size_valid() shared helper (mirrors face.rs check)
  so the pre-flight validation threshold stays in one place

gstate.rs:
- Expand mat2x2_mul doc to show which indices are used/dropped and
  the row-major 2x2 product layout
- Add three unit tests for mat2x2_mul: identity, scaling, and 90°
  rotation (regression guard for the Trm computation correctness)

page.rs:
- Remove spurious .clone() in MoveNextLineShow / MoveNextLineShowSpaced:
  bytes/text are &Vec<u8> from &Operator, not from self
- Add comment: trm is stable across the glyph loop (translation
  components don't affect the 2×2 submatrix)
- Add comment: pen advance applies even when make_glyph returns None
  (PDF §9.4.4 requirement)
- Fix render mode comment: modes 4-6 would paint + clip, not invisible;
  note this as a known limitation since clip accumulation is not yet impl
- Log debug when Tz 0 % override fires (silent before)
- Guard blit_image against zero-size images and non-finite CTM corners

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Implement DCTDecode (JPEG) and JPXDecode (JPEG 2000) image decoding

- DCTDecode via zune-jpeg: handles Luma (grayscale), RGB, and CMYK
  JPEGs; CMYK is converted to RGB via complement inversion
- JPXDecode via jpeg2k/OpenJPEG: handles all bit depths (8/16) and
  channel counts (L, La, RGB, RGBA), downscaling 16-bit to 8-bit via
  high-byte extraction
- Both decoders log a debug message when PDF dict dimensions differ from
  the decoded image dimensions and use the image's own dims as authoritative
- Stale do_xobject comment updated to reflect new filter support

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Hardening pass — logic, duplication, and idioms
- Eliminate duplication, extract shared helpers, split god files
- Hardening pass on refactor commit

dict_ext: drop Object import; use as_bool() instead of manual match for
get_bool (consistent with get_name/get_i64); downgrade trait from pub to
pub (redundant_pub_crate inside a pub(crate) module)

text_ops: GlyphRecord fields use plain pub on a pub(super) struct —
redundant pub(super) per-field removed (idiomatic)

testutil: rect_path uses .expect() with per-step messages instead of
.unwrap() so test failures identify which builder call failed

fill/parallel: strip redundant per-item #[cfg(feature = "rayon")]
attributes — the module is only compiled when the feature is set, making
all per-item cfg attrs dead weight; replace stale type-layout comment
with a concise aliasing-safety note

stroke/path: guard make_dashed_path against subnormal line_dash_total
(< f64::EPSILON instead of == 0.0) to prevent splash_floor overflow on
phase / line_dash_total with tiny non-zero totals

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Fix image SMask — skip images whose soft mask can't be decoded
- Harden SMask decoding — validation, overflow safety, correct defaults
- Implement Form XObject recursive execution
- Implement Encoding/Differences array for Type 1 / TrueType fonts
- Implement ExtGState opacity and graphics parameter overrides
- Refuse to open PDFs containing JavaScript
- Implement W/W* clip path operators
- Harden clip path implementation
- Add ROADMAP.md

Replaces the memory-only tracking with a proper in-repo roadmap.
Phase 1 (native interpreter) lists what's done and what's next.
Phases 2–4 cover raster perf, coverage completeness, and GPU work.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Split Phase 1 into blocking/nice-to-have/parking-lot, ordered by priority
- Add hardware context to Phase 2/4; corpus note on image decode dominance
- ICCBased/Indexed/CMYK colour spaces + fmt cleanup
- Hardening pass on colour space decoding
- Wire ExtGState blend modes (BM key)
- CCITTFaxDecode Group 3 1D (K=0) support
- Inline images (BI ID EI) + RunLengthDecode filter
- Shading (sh) — axial and radial gradients wired through
- Mark Phase 1 complete; update preamble to reflect native-only CLI
- Hardening pass: shading, inline image, and run-length decode

- shading/eval_stitching: validate bounds.len() == num_fns-1; log::warn
  and fall back to first sub-function on malformed Type 3 function dicts
  instead of panicking or silently accessing out-of-bounds index
- shading/eval_stitching: clamp idx to num_fns-1 as a second safety net
- shading/resolve_axial, resolve_radial: reject non-finite Coords before
  and after CTM transform (log::warn + return None)
- shading/ctm_scale: fall back to 1.0 on non-finite CTM (NaN/Inf)
- shading: upgrade ShadingType unsupported log from debug → warn so it
  surfaces in normal usage
- renderer/do_shading: upgrade missing-shading log from debug → warn
- image/decode_run_length: cap output at 256 MiB to prevent adversarial
  OOM from a crafted RunLengthDecode stream
- Tests: eval_stitching_wrong_bounds_count_falls_back,
  ctm_scale_nan_falls_back_to_one, ctm_scale_inf_falls_back_to_one,
  run_length_truncates_at_max_output (104 tests total, all passing)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Phase 2: FDot16 fixed-point edge stepping in scanner

Add `dxdy_fp: i32` (16.16 fixed-point) to `XPathSeg`, computed as
`(dxdy * 65536.0).round()` with i32-range saturation.

The scanner's sloped-segment inner loop now steps the right-edge
accumulator as `xx1_fp += dxdy_fp` (integer add per scanline) rather
than the previous `f64` add `xx1 += dxdy`.  The accumulator is still
initialised from `f64` for the first row to preserve accuracy; the
per-row hot path is integer-only.

Benefits:
- Eliminates f64 dependency chains in the scan-conversion inner loop
- Integer arithmetic is trivially vectorizable; f64 is not
- 1/65536 ≈ 1.5e-5 px/row precision — error accumulates to at most
  one pixel per ~65k scanlines, far beyond any real page height

Also mark Phase 2 items 1–3 complete in ROADMAP; only sparse tile
rasterisation remains.

4 new tests: dxdy_fp_matches_dxdy_for_slope_one, _half_slope,
_zero_for_horizontal, _zero_for_vertical (235 raster tests total).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
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
- Fix annotation BBox bug, transparency group hardening, serial test execution

Annotation rendering fix (critical):
- render_one_annotation was calling read_rect() (looks for Rect key) on an
  appearance stream dict that has BBox, not Rect — silently returning None
  and skipping all annotations.  Now uses form.bbox which form_from_stream_id
  already populated correctly.  Removes the redundant second doc.get_object()
  call entirely.

Transparency group hardening:
- Use and_then instead of map so a non-finite BBox/CTM falls back to rendering
  without a group rather than casting NaN to i32 (UB on some platforms).
- Capture parent fill_alpha and blend_mode before gstate.save() so the group
  is composited using the invoking context's opacity, not the form's final state.

Shared helper:
- Extract read_f64_n<const N> (pub(crate)) in resources/mod.rs; rewrite
  read_bbox and read_matrix as one-liners on top of it.  read_rect in
  page/mod.rs also delegates to it, eliminating duplicate parsing loops.
- read_bbox now normalises inverted axes (same as read_rect), preventing
  the debug_assert panic in begin_group on malformed PDFs.

Serial test execution:
- Add [test] test-threads = 1 to .cargo/config.toml.  Concurrent tests with
  multi-hundred-MiB working sets OOM the machine under cargo test --lib.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Cargo fmt

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Wire GPU tile fill into fill_path dispatch + refactor
- Mark Phase 4 item 3 fully complete
- Review pass on GPU fill helpers
- Bake ICCBased CMYK CLUT from moxcms for accurate colour conversion
- Add CLAUDE.md — crate map, build/test commands, OOM rules, GPU context

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Cargo fmt

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Mark Phase 2.5 complete; Phase 4 ICC item complete
- Implement bpc 2, 4, 16 image decoding
- Harden bpc 2/4/16 image decoding
- Implement CCITTFaxDecode K>0 (Group 3 mixed 2D / T.4 MR)
- Hardening pass: --gray/--mono and CCITTFax K>0

cli/render.rs:
- rgb_to_gray: replace manual x*3 indexing with chunks_exact(3) iterator
  to eliminate the implicit bounds-arithmetic; document the stride/width
  relationship; add width-cast comment
- gray_to_mono: slice to [..w] explicitly before zipping; document the
  50%-midpoint threshold convention matching pdftoppm
- collapse duplicate intermediate binding in mono path: gray_to_mono(&rgb_to_gray(&rgb))
- shorten identical unreachable! messages (same text three times)

encode/pbm.rs:
- remove spurious #[expect(cast_possible_truncation)] on u32→usize (clippy
  does not fire this lint for that cast); replace with brief inline comment

pdf_interp/src/resources/image.rs — HayroCcittCollector:
- finish() / next_line(): replace `(width - col) as usize` with
  `usize::try_from(…).unwrap_or(0)` so the intent is explicit and the
  safe-subtraction invariant is documented
- decode_ccitt_g3_2d: fix truncation bug — a malformed stream that emits
  more pixels than p.capacity would leave data_out.len() > p.capacity;
  the previous `resize` only extended, never truncated. Now:
  truncate first if over-long, then resize-with-white if short.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Implement Type 1 (function-based) shading via pre-sampled grid
- Hardening pass — BBox intersection, casts, mul_add, dedup
- Record Apr 2026 benchmark results and /Rotate gap
- Fix /Rotate and CropBox page geometry
- Phase 0 complete, Phase 5 deskew algorithm decided
- New library crate — raster_pdf API + deskew foundation
- Review pass: hardening, logic, idioms, edge cases, docs

gpu/nvjpeg2k.rs:
- Add NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED constant (was magic 9 in comments)
- Remove redundant debug_assert_eq! on structurally-guaranteed comp_infos.len()
- Promote debug_assert_eq! in copy_and_interleave to assert! — cheap per-decode,
  guards against silent data corruption from future refactoring
- swap_remove(0) instead of remove(0) for Gray plane (no-op perf, correct idiom)
- Tighten NvJpeg2k::Drop comment: null checks are defence-in-depth only

gpu/shim/nvjpeg2k_shim.cpp:
- Drop magic number "(9)" from header comment; constant name is self-documenting

pdf_raster/render.rs:
- Replace PageIter sentinel hack (Deskew(String::new()) placeholder) with
  Option<Result<RenderState, RasterError>> — exhaustion and error yield are
  now structurally distinct, not distinguished by a never-read sentinel value
- Add InvalidOptions error variant: validate dpi > 0, first_page >= 1,
  first_page <= last_page before opening the document; callers get a clear
  error message instead of silent empty output or a confusing downstream panic
- Cache doc.get_pages() in RenderState (built once): eliminates O(n²) cost
  from calling it per-page on multi-page documents

pdf_raster/lib.rs:
- Document dpi > 0, first_page >= 1, last_page >= first_page constraints
- Doc example: replace .expect() with match to show proper error handling
- Add InvalidOptions to raster_pdf # Errors list

pdf_raster/deskew/rotate.rs:
- Inline bilinear() into rotate_cpu(): removes per-pixel function-call overhead
  and allows the compiler to eliminate the redundant w/h loads from Bitmap
- Precompute sx_max/sy_max bounds from w/h once per frame instead of per-pixel
- Fix bounds check: x0 > sx_max (where sx_max = w-2) is semantically identical
  to the old x0+1 >= w but avoids the addition and clarifies the 2×2 requirement
- Update module doc: note the 2×2 neighbourhood constraint on the last valid origin

pdf_raster/deskew/detect.rs:
- assert!(factor > 0) in downsample() — factor=0 would divide by zero; the
  constant DOWNSAMPLE=4 can never trigger this but explicit is better than silent

ROADMAP.md: mark completed work items, add remaining GPU rotation TODO

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Mark CLI thin-wrapper item complete
- Review pass 2: harden render pipeline and CLI output safety

- render_page_rgb: reject non-finite/non-positive scale before any page I/O
- open_session: log GPU init failure with actionable hint instead of silently
  dropping it
- PageIter::next Err arm: remove tautological .expect; use take()?.map()
- bitmap_to_vec: remove unused h variable
- lib.rs: correct # Panics doc (GPU driver bugs, not normal operation);
  fix doc example unwrap() → expect()
- CLI render_page: extract MONO_THRESHOLD const (128) and reference it in
  gray_to_mono; write to .tmp then rename atomically so a failed encode never
  leaves a partial output file at the final path
- ROADMAP: mark second review pass complete

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Review pass 3: correctness, idioms, and hardening across library and CLI

render.rs (library):
- open_session: derive total_pages from the pages map (was calling get_pages twice)
- render_page_rgb: invalid scale returns InvalidOptions, not PageDegenerate
- PageIter::next Err arm: replace map(|_|unreachable!()) with explicit match
- Sync assertion: compile-time const fn verifies RasterSession: Sync
- #[expect] reasons corrected: remove erroneous cast_sign_loss on f64→f32

deskew/detect.rs:
- rayon import moved to module level
- Coarse sweep: eliminate two intermediate Vec allocations; use par_iter reduce
- SWEEP_STEPS >= 2 compile-time assert (infinite loop guard)
- n_rows - skip: saturating_sub + debug_assert guard
- assert! → debug_assert! in private downsample
- Remove false AVX-512 auto-vec claim (scatter writes are not vectorisable)
- #[expect] cast_sign_loss removed for f32→i32 (lint does not apply to this cast)
- Rename misleading test; fix refine/binary-search terminology in doc

deskew/rotate.rs:
- Correct "CCW positive" → "CW positive" throughout (the matrix is CW-positive;
  two matching sign conventions made deskew work correctly but docs were wrong)
- Fix Vec OOM comment: panics, not aborts

cli/args.rs:
- #[allow] → #[expect] on struct_excessive_bools with reason
- DPI args (−r, −−rx, −−ry): validated ≥ 1 at the CLI boundary
- jpeg_quality: validated in range 0..=100
- OutputFormat implements Display; error messages use {} not {:?}
- Remove 13 redundant default_value_t = false on bool fields

cli/main.rs:
- --odd and --even mutual exclusion enforced with early exit
- open_session error now walks the full source chain (was printing top level only)

cli/render.rs:
- fs::rename failure path also removes the temp file

lib.rs:
- deskew field doc: note GPU rotation is not yet implemented (was stated as done)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
- Mark GPU deskew rotation complete
- Hardening, idiom, and logic review pass
- Close Phase 5, open Phase 6
- Implement UserUnit scaling and effective_dpi
- Review pass: harden UserUnit, dedup page-ID lookup, fix docs

- Extract resolve_page_id() helper used by both page_size_pts and
  parse_page, eliminating the double get_pages() call (and double
  bounds check) in each function.

- UserUnit validation: check is_finite() before the range test so that
  a NaN/Inf Real object (which passes NaN comparisons silently) is
  caught; also detect and report non-numeric UserUnit objects using
  lopdf's enum_variant() for a clear type-error message instead of a
  misleading "0 outside [0.1, 10.0]".

- Fix render_page_rgb # Errors doc: scale ≤ 0 now returns InvalidOptions
  (not PageDegenerate as the old doc said); add InvalidPageGeometry entry.

- Fix render_one effective_dpi comment: UserUnit:2.0 doubles effective DPI
  (larger physical units → more DPI per pixel), not halves it.

- Fix #[expect] reason on effective_dpi cast: state concrete value bounds
  (dpi ≤ ~3400, user_unit ≤ 10 → product ≤ ~34 000, well within f32).

- Fix RenderedPage.dpi doc: remove circular cross-reference to
  effective_dpi; clarify that effective_dpi is always the right value
  to pass downstream.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
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


