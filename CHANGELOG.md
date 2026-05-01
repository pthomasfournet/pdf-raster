# Changelog

All notable changes to this project will be documented in this file.

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


