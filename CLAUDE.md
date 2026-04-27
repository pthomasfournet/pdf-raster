# pdf-raster — Claude Code guide

Full Rust PDF → pixels pipeline. Zero poppler in the render path.

## Crate map

| Crate | Role |
|---|---|
| `crates/color` | Pixel types, colour math |
| `crates/raster` | Pixel-level renderer (~9 000 lines) |
| `crates/font` | FreeType glyph cache + rendering |
| `crates/encode` | PPM/PNG/PGM output |
| `crates/pdf_interp` | Native PDF interpreter — the only CLI render path |
| `crates/pdf_bridge` | poppler C++ wrapper — retained as reference baseline, **not linked by CLI** |
| `crates/cli` | Binary entry point |
| `crates/gpu` | CUDA kernels via cudarc 0.19 (Porter-Duff, soft mask, nvJPEG, AA fill, tile fill, ICC CLUT) |
| `crates/bench` | Synthetic fill benchmark vs vello_cpu |

## Build

```bash
# Check (preferred — no artifacts)
cargo check -p pdf_interp

# Release binary
cargo build --release -p cli

# With GPU features
cargo build --release -p cli --features "pdf_interp/nvjpeg,pdf_interp/gpu-aa,pdf_interp/gpu-icc"

# PTX compilation uses NVCC; target arch via env var (default sm_80):
CUDA_ARCH=sm_120 cargo build --release -p gpu
```

## Testing

```bash
# Unit tests — always filter, never run --lib unfiltered
cargo test -p gpu --lib -- icc
cargo test -p pdf_interp --lib -- resources

# Pixel-diff comparison vs pdftoppm
tests/compare/compare.sh -r 150 tests/fixtures/ritual-14th.pdf

# Fill microbenchmark
RUSTFLAGS="-C target-cpu=native" cargo run -p bench --release -- --iters 30 --stars 200
```

## OOM / disk rules (30 GB RAM machine)

1. `df -h /` before any `cargo build/check/test` — abort if < 20 GB free.
2. Never more than 3 compilation commands back-to-back without re-checking disk.
3. **Never `cargo build` when `cargo check` suffices.**
4. After `cargo check`, delete `target/debug/` to reclaim space.
5. Prefer `-p <crate>` over workspace-wide builds.
6. **Never `--all-targets` in clippy** — compiles test binaries and OOMs.
7. `#[test]` needing tokio: use `#[tokio::test]` or `Builder::new_current_thread()`, never `Runtime::new()`.
8. After `drain.abort()`, always `let _ = drain.await`.

## Hardware context

- CPU: Ryzen 9 9900X3D — AVX-512 (`avx512f/bw/vl/dq/cd/ifma/vbmi/vbmi2/vnni/bf16/bitalg/vpopcntdq`), 128 MB 3D V-Cache. Build with `-C target-cpu=native`.
- GPU: RTX 5070, CC 12.0 Blackwell, 12 GB GDDR7. PTX target `sm_120`.

## GPU dispatch thresholds

| Path | Threshold | Constant |
|---|---|---|
| nvJPEG | ≥ 512×512 px | `GPU_JPEG_THRESHOLD_PX` |
| GPU AA fill | ≥ 16 384 px | `GPU_AA_FILL_THRESHOLD` |
| Tile fill | ≥ 65 536 px | `GPU_TILE_FILL_THRESHOLD` |
| ICC CMYK→RGB | ≥ 500 000 px | `GPU_ICC_CLUT_THRESHOLD` |

`fill_path` dispatch order: tile fill → AA fill → CPU scanline AA.

## Lints

Workspace enforces `unsafe_code = "deny"` on all crates except `gpu`. Use `#[expect(lint, reason = "...")]` not `#[allow]`. Never suppress `unsafe_code` in `pdf_interp`.

## Roadmap status

See `ROADMAP.md`. Phases 1–3 complete. Phase 4 (GPU) active:
- nvJPEG ✓, tile fill ✓, ICC CMYK→RGB ✓, GPU AA kernel ✓
- **Open:** GPU AA quality validation (pixel-diff vs CPU AA), avx512vnni ICC matrix, threshold tuning
