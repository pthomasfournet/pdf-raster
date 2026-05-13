# Parallel Huffman JPEG Decode on GPU: A Post-Mortem

We spent roughly two weeks implementing a custom parallel Huffman JPEG decoder that runs on both CUDA and Vulkan compute shaders. The short version: **it works, it's correct, and it mostly doesn't beat the CPU.** That was the expected outcome going in — and we did it anyway. This page is the story of why, what we learned, and what the numbers actually look like.

---

## Why we built it

Standard JPEG decode is inherently serial. The Huffman entropy stage processes one variable-length codeword at a time; each codeword's bit-position depends on all previous codewords. You cannot send page 1 and page 2 of the bitstream to two threads and let them race. This is the hard problem.

In 2018, Weißenberger and Schmidt published a self-synchronizing parallel Huffman algorithm. The core idea: split the bitstream into equal-size substreams, have each GPU thread independently decode a substream from a guessed start state, then use a linear-scan synchronization pass to stitch the pieces together once the "phase lock" point — the first bit boundary where two adjacent decoders agree — is identified. Wei et al. (2024) extended this to JPEG framing and reported 51× speedup over libjpeg-turbo on A100.

That last sentence is the bait. **51× on A100 vs single-threaded libjpeg-turbo is a very different comparison from the one we were actually running:** consumer Blackwell (RTX 5070) or consumer Ada (RTX 4080) vs 24-thread zune-jpeg with AVX-512 IDCT.

We knew this going in. The decision to build it anyway was explicit and recorded in ROADMAP.md:

> The Weißenberger 2018 algorithm is genuinely beautiful CUDA work. Implementing it produces an open, redistributable artifact that demonstrates a non-obvious algorithm. The work has long-term value as a learning project; it's just not the path to faster rasterrocket.

This is an academic post-mortem, not a performance story. The algorithm is interesting. We wanted to understand it from the inside.

---

## The algorithm

JPEG Huffman decode is a context-sensitive scan — the decoder maintains a predictor state (DC coefficient of the previous MCU, per component) and the codeword boundaries are only known by decoding from the beginning. Parallel decode has to solve two sub-problems:

1. **Phase lock:** given a mid-stream start position, decode forward until you're confident you're aligned to a real codeword boundary. The key insight: a short run of consecutive successful Huffman table lookups, combined with the structure of JPEG's MCU schedule (each MCU decodes exactly `H × V × (1 + num_AC)` codewords per component), is strong probabilistic evidence of alignment. The Wei algorithm calls this intra-subsequence synchronization.

2. **DC predictor inheritance:** once each substream knows its own relative DC values, you need the absolute offset from the first MCU. This is a prefix-sum problem — a Blelloch scan on the delta-DC values from each substream gives every substream its final DC base in O(log n) passes on the GPU.

The four phases:

- **Phase 1 — intra-sync:** each thread decodes its substream from the start, recording the first point where the MCU schedule closes cleanly. This is the "candidate alignment." Boundary-snapshot semantics matter here: the snapshot captures the decoder state at the first crossing into the *next* substream's region, not the over-walk state, so Phase 4 inheritance composes cleanly.
- **Phase 2 — inter-sync (bounded retry):** verify that adjacent substreams agree on their boundary. Disagreement means the Phase 1 candidate was a false lock; retry with a search window. Bounded at `2 × log₂(n)` retries. Failures surface as a typed `SyncBoundExceeded` error — never a hang or wrong output.
- **Phase 3 — Blelloch scan:** prefix-sum over the per-substream DC deltas to compute absolute DC offsets.
- **Phase 4 — re-decode + write:** each thread re-decodes its substream from its confirmed start point, applying the Phase 3 DC base, and writes coefficients to the output buffer.

---

## Implementation

The implementation lives in `crates/gpu/src/jpeg/`. The algorithm runs on both CUDA (PTX, `sm_80`+) and Vulkan compute (Slang, compiled to SPIR-V at build time). Both backends produce byte-identical output, verified by a 10-vector adversarial corpus in `huffman/corpus.rs` covering:

- Short / long / uniform / skewed symbol distributions
- Single-symbol alphabets (degenerate case)
- Max-length-16 Huffman codes
- Phase-2 retry triggers (crafted false-lock sequences)
- One-subseq degenerate case
- Word-aligned and max-tail-padding boundaries

The CUDA path uses `cooperative_groups::sync` for intra-phase barriers; the Slang path uses `AllMemoryBarrierWithGroupSync()`. Both are invoked from the same dispatch layer in `crates/gpu/src/jpeg/dispatcher.rs`.

We wired the path into the production decode pipeline behind `GPU_JPEG_HUFFMAN_THRESHOLD_PX`. It's enabled per-image when the pixel count exceeds the threshold. In production the threshold is `u32::MAX` — the path is dormant. You can force it for benchmarking with `PDF_RASTER_HUFFMAN_THRESHOLD=0`.

---

## The numbers

These are from the v0.9.1 five-mode benchmark. Mode A is CPU-only (`zune-jpeg`, AVX-512). Mode C-std is CUDA+nvJPEG. Mode C-huff is CUDA parallel-Huffman (`PDF_RASTER_HUFFMAN_THRESHOLD=0`). Mode V-huff is Vulkan parallel-Huffman.

All numbers are mean ± σ from 5 hyperfine runs with cold-cache eviction between runs, on AMD Ryzen 9 9900X3D + RTX 5070, Linux 6.17.

| Corpus | A. CPU-only | C-huff / A | V-huff / A | C-huff vs C-std |
|---|---|---|---|---|
| 01 native text, small (16 pp) | 41 ± 1 ms | **5.9×** slower | **35×** slower | +6% |
| 02 native vector + text (16 pp) | 18 ± 1 ms | **13×** slower | **77×** slower | −53% |
| 03 native text, dense (254 pp) | 231 ± 2 ms | **1.8×** slower | **9.4×** slower | −50% |
| 04 ebook, mixed (358 pp) | 278 ± 3 ms | **1.7×** slower | **8.0×** slower | −36% |
| 05 academic book (601 pp) | 582 ± 12 ms | **1.4×** slower | **4.0×** slower | −1% |
| 06 modern layout, DCT (160 pp) | 1 450 ± 10 ms | **1.1×** slower | **2.8×** slower | −1.5% |
| 07 journal, DCT-heavy (162 pp) | 783 ± 5 ms | **1.2×** slower | **2.8×** slower | −20% |
| 08 1927 scan, DCT baseline (390 pp) | 1 652 ± 89 ms | **1.2×** slower | **1.7×** slower | −10% |
| 09 1836 scan, DCT progressive (490 pp) | 2 859 ± 658 ms | **0.83×** ✓ faster | **1.1×** slower | −10% |
| 10 scan, JBIG2+JPX (576 pp) | 17 616 ± 260 ms | **1.0×** (neutral) | **1.0×** (neutral) | +2% |

**CPU wins on 9 of 10 corpora.** The GPU Huffman path beats CPU-only exactly once — corpus 09, progressive JPEG, 490 pages — and that result has a σ of 658 ms (23% of mean) due to background load, making it unreliable.

The interesting comparison is **C-huff vs C-std (nvJPEG)**. On text-heavy corpora (01–04), C-huff is 36–53% faster than nvJPEG. This is not because the parallel-Huffman decoder is faster at Huffman decoding — it's because nvJPEG on consumer hardware has significant per-image dispatch overhead that the custom path doesn't. On JPEG-heavy corpora (06–10) the margin closes to ±10%.

The comparison against CPU is the honest one. A 12-core/24-thread CPU with AVX-512 running `zune-jpeg` is a 24-way parallel decoder with SIMD-optimized IDCT. Our GPU decoder is also parallel, but it has PCIe latency and dispatch overhead that the CPU doesn't. For the aggregate-throughput workload that rasterrocket runs — many independent JPEG images, all pages of a document, 24 threads — the CPU wins because it keeps all 24 cores saturated without any PCIe round-trips.

The A100 result in the paper (51× speedup) is a different workload: a **single** large JPEG image, measured against **single-threaded** libjpeg-turbo. That scenario does not appear in rasterrocket's actual usage.

---

## What the numbers mean for real workloads

The only way GPU parallel-Huffman could win in aggregate is if the CPU were the bottleneck — i.e., you have more JPEG images to decode than CPU cores to decode them, and the images are large enough that GPU dispatch amortizes. That scenario exists in datacenter inference workloads (single-machine, hundreds of concurrent documents, no spare threads). It does not match a 24-thread desktop machine running a 490-page scan corpus.

For **single-page latency** on a scan-heavy page, there's a plausible win: the GPU decodes one large JPEG in parallel while the CPU is busy with the rest of the render pipeline. We did not characterize this scenario separately because it isn't the current dispatch model.

The dormant threshold (`u32::MAX`) stays dormant until we run a proper single-page latency experiment and find an image size where dispatch overhead is dominated by decode time.

---

## What we learned

**The algorithm is harder to implement than it looks.** The Weißenberger paper fits in 8 pages. The implementation took several weeks and generated a dozen hardening commits after the initial "it works" milestone. The key correctness subtleties:

- **Boundary-snapshot semantics in Phase 1.** `s_info[i]` must capture state at the *first decoding step that crosses into subseq (i+1)'s region*, not the over-walk end state. The over-walk state loses information about what the aligned decoder would have seen. Getting this wrong produces a decoder that passes most tests but fails on corpora where Phase 4's DC inheritance reverses direction.
- **The Phase 2 retry bound isn't always achievable.** Mixed-codeword-length corpora and multi-component streams with DC-delta rollover can hit the `2 × log₂(n)` bound. We surface this as `SyncBoundExceeded`, never as a hang or silently wrong output. Production JPEG streams rarely trigger this; the adversarial corpus does.
- **Synthetic test fixtures have constraints real JPEG doesn't.** The kernel advances `code_bits + value_bits` per decode step. For a uniform-code-length table this means all symbols still have different advances if their low nibbles (value-bit counts) vary. Synthetic fixtures must pick symbols with `symbol & 0x0F == 0` to get uniform advances. Real JPEG AC tables don't have this constraint (the low nibble is the AC value bit count, which the MCU schedule knows independently), but it confused the fixture builder for a while.
- **GPU backend differences are real.** CUDA's `cooperative_groups::sync` and Slang's `AllMemoryBarrierWithGroupSync()` have different barrier semantics with respect to shared memory visibility. Getting byte-identical output on both required careful audit of the barrier placement in Phase 1 and Phase 4.

**The perf model for GPU JPEG is often cited wrong.** The 51× number lives in GPU JPEG conversations like a consensus claim. It's a real measurement, but it assumes single-threaded CPU baseline and datacenter GPU. The honest framing for consumer workloads is: GPU parallel-Huffman is roughly competitive with CPU parallelism for very large individual images (> ~4 MP), slower for everything else when you account for dispatch overhead, and much slower than nvJPEG for typical PDF JPEG images (typically 1–8 MP) where the fixed-function decoder is already fast enough.

**Building something you expect to lose is instructive.** We had a clear hypothesis going in: 24-thread AVX-512 CPU would win. The implementation confirmed it. More importantly, it confirmed *why* it wins — the PCIe round-trip and dispatch overhead are not negligible at this image size distribution — and gave us a working, tested GPU decoder that can be enabled trivially if the workload ever changes.

---

## Status

The parallel-Huffman path is **wired into production** and **dormant by default**. It runs on both CUDA and Vulkan. It produces byte-identical output to the CPU path. It is tested with an adversarial corpus. It is not automatically dispatched.

To enable it for a benchmark or experiment:

```bash
PDF_RASTER_HUFFMAN_THRESHOLD=0 rrocket --backend cuda input.pdf out
PDF_RASTER_HUFFMAN_THRESHOLD=0 rrocket --backend vulkan input.pdf out
```

The source is in `crates/gpu/src/jpeg/` — `huffman.rs` (algorithm), `bitreader.rs` (shared CPU oracle and GPU pre-pass), `dispatcher.rs` (dispatch routing), `corpus.rs` (adversarial test vectors).

---

## References

- Weißenberger, A. and Schmidt, B. (2018). *Massively Parallel Huffman Decoding on GPUs*. In Proceedings of the 47th International Conference on Parallel Processing (ICPP '18). ACM.
- Wei, J. et al. (2024). *Massively Parallel JPEG Decoding on GPU*. Preprint.
