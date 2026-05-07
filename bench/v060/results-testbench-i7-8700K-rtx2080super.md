# v0.6.0 GPU baseline — testbench (i7-8700K + RTX 2080 SUPER)

**Machine:** Intel Core i7-8700K (Coffee Lake, 6c/12t, 3.7 GHz base / 4.7 GHz boost,
12 MB L3, AVX2 only — no AVX-512), 30 GB RAM, RTX 2080 SUPER (Turing CC 7.5,
8 GB GDDR6, 3072 CUDA cores), Ubuntu 25.10, CUDA 12.4.

**Why this matrix exists:** to test the hypothesis "users with weaker CPUs paired
with their GPU would benefit from nvJPEG dispatch even though our 9900X3D doesn't".
The data shows the hypothesis is wrong: GPU dispatch via `GPU_HYBRID` loses on
this configuration too, for the same root cause (per-call CUDA API overhead +
CPU Huffman behind a CUDA wrapper).

**Differences from the 9900X3D primary baseline:**
- nvJPEG2000 not installed (libnvjpeg2k.so unavailable on Ubuntu 25.10 stock CUDA),
  so corpus 10 ran the CPU JPEG2000 path on every mode — the GPU advantage on
  corpus 10 visible on the primary machine disappears here.
- PTX target arch sm_75 (Turing) instead of sm_120 (Blackwell). All kernels built
  successfully.
- 12 threads instead of 24, no AVX-512.

Cells show `mean±stddev` in milliseconds. `tests/bench_corpus.sh --backend X`
+ hyperfine `--warmup 1 --runs 5` + `posix_fadvise(FADV_DONTNEED)` cache eviction.

| Corpus | A. CPU-only | C. nvJPEG | D. Full GPU | Flags |
|---|---|---|---|---|
| 01 native text small | 192±102ms | 256±90ms | 541±17ms | A:[high-σ 53%] C:[high-σ 35%] |
| 02 native vector+text | 126±4ms | 520±46ms | 489±47ms |  |
| 03 native text dense | 673±111ms | 935±56ms | 1048±103ms | A:[high-σ 16%] |
| 04 ebook mixed | 1157±100ms | 1276±18ms | 1499±86ms |  |
| 05 academic book | 3735±30ms | 3883±30ms | 4383±501ms |  |
| 06 modern layout DCT | 10860±62ms | 10939±23ms | 11568±919ms |  |
| 07 journal DCT heavy | 1162±36ms | 1413±101ms | 1507±43ms |  |
| 08 scan DCT 1927 (baseline JPEG) | 4805±79ms | 4972±63ms | 5270±311ms |  |
| 09 scan DCT 1836 (progressive JPEG) | 6200±230ms | 6377±101ms | 6271±113ms |  |
| 10 scan JBIG2+JPX | 58618±574ms | 59374±730ms | 57946±287ms |  |

## Verdict

GPU mode C is **never faster** than mode A on this machine. Best case: tied
(corpora 06/08/09/10 are within stddev). Worst case: 4× slower (corpus 02,
small page where per-call CUDA overhead dominates).

This confirms the v0.6.0 threshold workaround (`GPU_JPEG_THRESHOLD_PX = u32::MAX`,
disabling nvJPEG dispatch) is correct on consumer hardware regardless of CPU
class. The conditional-by-hardware-class design that was being considered is
not justified by the data.

The next architectural fix that could change this picture is page-level batched
nvJPEG submission (`nvjpegDecodeBatched`) — see
`docs/superpowers/specs/2026-05-06-batched-nvjpeg.md`. The bench-gate criterion
in that spec stands: ship batched only if it gives a strict win on at least
one DCT-heavy corpus.
