//! AA pipe: shape byte present, `BlendMode::Normal`, no soft mask, isolated group.
//!
//! Equivalent to `Splash::pipeRunAA{Mono8,RGB8,XBGR8,BGR8,CMYK8,DeviceN8}`.
//!
//! For each pixel:
//! 1. `a_src = div255(a_input * shape)` — scale source alpha by AA coverage.
//! 2. If `a_src == 255`: direct write (no read-back needed).
//! 3. If `a_src == 0` and `a_dst == 0`: write zeros.
//! 4. Otherwise: `a_result = a_src + a_dst - div255(a_src * a_dst)`.
//!    `c_result = ((a_result - a_src) * c_dst + a_src * c_src) / a_result`.
//!    Then apply transfer LUT.

use std::cell::RefCell;

use crate::pipe::{self, PipeSrc, PipeState};
use crate::simd::composite_aa_rgb8_opaque;
use crate::types::BlendMode;
use color::Pixel;
use color::convert::div255;

// Per-thread scratch buffer for pattern spans — grow-never-shrink, zero per-span alloc.
thread_local! {
    static PAT_BUF: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Composite a span with per-pixel shape (AA coverage) bytes.
///
/// `shape[i]` is the coverage for pixel `x0 + i`.  Length must equal
/// `x1 - x0 + 1`.
///
/// # Preconditions (checked in `render_span`)
///
/// - `pipe.use_aa_path()` — no soft mask, `BlendMode::Normal`, no group correction.
/// - `dst_pixels.len() == count * P::BYTES`.
/// - `shape.len() == count`.
/// - `P::BYTES > 0`.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors C++ SplashPipe API; all parameters are necessary"
)]
pub(crate) fn render_span_aa<P: Pixel>(
    pipe: &PipeState<'_>,
    src: &PipeSrc<'_>,
    dst_pixels: &mut [u8],
    dst_alpha: Option<&mut [u8]>,
    shape: &[u8],
    x0: i32,
    x1: i32,
    y: i32,
) {
    debug_assert_eq!(pipe.blend_mode, BlendMode::Normal);
    debug_assert!(pipe.soft_mask.is_none());

    #[expect(
        clippy::cast_sign_loss,
        reason = "x1 >= x0 is a precondition, so x1 - x0 + 1 >= 1 > 0"
    )]
    let count = (x1 - x0 + 1) as usize;
    let ncomps = P::BYTES;

    debug_assert_eq!(shape.len(), count, "shape length must equal pixel count");
    debug_assert_eq!(dst_pixels.len(), count * ncomps);

    let a_input = u32::from(pipe.a_input);

    match src {
        PipeSrc::Solid(color) => {
            debug_assert_eq!(color.len(), ncomps);

            // Fast path: solid RGB source, no alpha plane, identity transfer.
            // composite_aa_rgb8_opaque processes 16 pixels/iter via [u16;16] lanes
            // that LLVM auto-vectorizes into AVX2/AVX-512.
            if dst_alpha.is_none() && ncomps == 3 && pipe.transfer.is_identity_rgb() {
                composite_aa_rgb8_opaque(
                    dst_pixels,
                    [color[0], color[1], color[2]],
                    pipe.a_input,
                    shape,
                );
                return;
            }

            // General solid path: read colour directly — no allocation.
            render_span_aa_inner(
                pipe,
                |_i| color,
                dst_pixels,
                dst_alpha,
                shape,
                count,
                ncomps,
                a_input,
            );
        }
        PipeSrc::Pattern(pat) => {
            // Reuse the thread-local scratch buffer — one allocation ever per thread,
            // grown as needed, never shrunk.
            PAT_BUF.with(|cell| {
                let mut buf = cell.borrow_mut();
                buf.resize(count * ncomps, 0);
                pat.fill_span(y, x0, x1, &mut buf[..count * ncomps]);
                render_span_aa_inner(
                    pipe,
                    |i| &buf[i * ncomps..(i + 1) * ncomps],
                    dst_pixels,
                    dst_alpha,
                    shape,
                    count,
                    ncomps,
                    a_input,
                );
            });
        }
    }
}

/// Inner AA compositing loop.
///
/// `src_px_at(i)` returns a `&[u8]` of length `ncomps` for the source pixel at
/// index `i`.  For solid sources this is always the same slice; for patterns it
/// indexes into the pre-filled scratch buffer.  Using a closure rather than a
/// `bool` flag keeps a single code path and lets the compiler inline both variants.
#[inline]
#[expect(
    clippy::too_many_arguments,
    reason = "all params necessary; closure eliminates the solid/pattern duplication"
)]
fn render_span_aa_inner<'src>(
    pipe: &PipeState<'_>,
    src_px_at: impl Fn(usize) -> &'src [u8],
    dst_pixels: &mut [u8],
    dst_alpha: Option<&mut [u8]>,
    shape: &[u8],
    count: usize,
    ncomps: usize,
    a_input: u32,
) {
    match dst_alpha {
        Some(dst_alpha) => {
            debug_assert_eq!(dst_alpha.len(), count);
            for i in 0..count {
                let shape_v = u32::from(shape[i]);
                let a_src = u32::from(div255(a_input * shape_v));
                let a_dst = u32::from(dst_alpha[i]);

                let (a_result, fully_opaque_src) = if a_src == 255 {
                    (255u32, true)
                } else if a_src == 0 && a_dst == 0 {
                    // Transparent src over transparent dst: zero and skip.
                    let base = i * ncomps;
                    dst_pixels[base..base + ncomps].fill(0);
                    dst_alpha[i] = 0;
                    continue;
                } else {
                    let ar = a_src + a_dst - u32::from(div255(a_src * a_dst));
                    (ar, false)
                };

                let base = i * ncomps;
                let src_px = src_px_at(i);
                let dst_px = &mut dst_pixels[base..base + ncomps];

                if fully_opaque_src {
                    // Full coverage: transfer src directly, no blending needed.
                    pipe::apply_transfer_pixel(pipe, src_px, dst_px);
                } else {
                    // Partial coverage: Porter-Duff over, then apply transfer.
                    for j in 0..ncomps {
                        let c_src = u32::from(src_px[j]);
                        let c_dst = u32::from(dst_px[j]);
                        // ((a_result - a_src) * c_dst + a_src * c_src) / a_result
                        let blended = ((a_result - a_src) * c_dst + a_src * c_src) / a_result;
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "blended = weighted average of values ≤ 255, divided by a_result ≤ 255"
                        )]
                        {
                            dst_px[j] = blended as u8;
                        }
                    }
                    pipe::apply_transfer_in_place(pipe, dst_px);
                }
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "a_result = a_src + a_dst - div255(a_src*a_dst) ≤ 255"
                )]
                {
                    dst_alpha[i] = a_result as u8;
                }
            }
        }
        None => {
            // No separate alpha plane: a_dst is implicitly 0xFF, a_result = 0xFF.
            // Formula simplifies to: c = div255((255 - a_src) * c_dst + a_src * c_src).
            for (i, &sh) in shape.iter().enumerate() {
                let shape_v = u32::from(sh);
                let a_src = u32::from(div255(a_input * shape_v));
                let base = i * ncomps;
                let src_px = src_px_at(i);
                let dst_px = &mut dst_pixels[base..base + ncomps];
                for j in 0..ncomps {
                    let blended =
                        div255((255 - a_src) * u32::from(dst_px[j]) + a_src * u32::from(src_px[j]));
                    dst_px[j] = blended;
                }
                pipe::apply_transfer_in_place(pipe, dst_px);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipe::PipeSrc;
    use crate::state::TransferSet;
    use color::{Rgb8, TransferLut};

    fn aa_pipe() -> PipeState<'static> {
        PipeState {
            blend_mode: BlendMode::Normal,
            a_input: 255,
            overprint_mask: 0xFFFF_FFFF,
            overprint_additive: false,
            transfer: TransferSet::identity_rgb(),
            soft_mask: None,
            alpha0: None,
            knockout: false,
            knockout_opacity: 255,
            non_isolated_group: false,
        }
    }

    #[test]
    fn full_coverage_writes_src() {
        let pipe = aa_pipe();
        let color = [200u8, 100, 50];
        let src = PipeSrc::Solid(&color);
        let shape = [255u8, 255];

        let mut dst = vec![50u8; 6]; // two pixels, initially different from src
        let mut alpha = vec![128u8; 2];

        render_span_aa::<Rgb8>(&pipe, &src, &mut dst, Some(&mut alpha), &shape, 0, 1, 0);

        assert_eq!(&dst[0..3], &[200, 100, 50]);
        assert_eq!(&dst[3..6], &[200, 100, 50]);
        assert_eq!(alpha[0], 255);
        assert_eq!(alpha[1], 255);
    }

    #[test]
    fn zero_coverage_over_transparent_zeroes_output() {
        let pipe = aa_pipe();
        let color = [255u8, 255, 255];
        let src = PipeSrc::Solid(&color);
        let shape = [0u8];

        let mut dst = vec![0u8; 3];
        let mut alpha = vec![0u8; 1]; // dest also transparent

        render_span_aa::<Rgb8>(&pipe, &src, &mut dst, Some(&mut alpha), &shape, 0, 0, 0);

        assert_eq!(dst[0], 0);
        assert_eq!(alpha[0], 0);
    }

    #[test]
    fn half_coverage_blends_correctly() {
        let pipe = aa_pipe();
        // src = white (255,255,255), dst = black (0,0,0), shape ≈ 128 ≈ 50%.
        let color = [255u8, 255, 255];
        let src = PipeSrc::Solid(&color);
        let shape = [128u8];

        let mut dst = vec![0u8; 3];
        let mut alpha = vec![255u8; 1]; // fully opaque destination

        render_span_aa::<Rgb8>(&pipe, &src, &mut dst, Some(&mut alpha), &shape, 0, 0, 0);

        // a_src = div255(255 * 128) ≈ 128.
        // a_result = 128 + 255 - div255(128 * 255) ≈ 255.
        // c = ((255 - 128) * 0 + 128 * 255) / 255 ≈ 128.
        let v = dst[0];
        assert!((125..=131).contains(&v), "expected ~128, got {v}");
        assert_eq!(alpha[0], 255);
    }

    #[test]
    fn no_alpha_plane_uses_opaque_dst() {
        let pipe = aa_pipe();
        let color = [200u8, 100, 50];
        let src = PipeSrc::Solid(&color);
        let shape = [128u8];

        let mut dst = vec![0u8; 3]; // black dst

        render_span_aa::<Rgb8>(&pipe, &src, &mut dst, None, &shape, 0, 0, 0);

        // With implicit a_dst=255: result should be a blend.
        // Expected: div255((255 - 128) * 0 + 128 * 200) ≈ 100.
        let v = dst[0];
        assert!((95..=105).contains(&v), "expected ~100, got {v}");
    }

    /// `TransferSet::is_identity_rgb()` gates a SIMD-friendly fast path
    /// (`composite_aa_rgb8_opaque`) that intentionally skips transfer-LUT
    /// application. If the predicate mis-reports `true` for a non-identity
    /// LUT (cargo-mutants whole-body → `true` survives without this test),
    /// the fast path runs and silently drops the transfer.
    ///
    /// Construct a non-identity LUT (channel-inverting), run `render_span_aa`,
    /// and require the inversion to be visible — only the general path
    /// applies it.
    #[test]
    fn non_identity_transfer_must_use_general_path() {
        // Inverting RGB transfer + identity gray/cmyk/device_n; the inverting
        // table is what makes this test's transfer set non-identity.
        static DN_ID: [[u8; 256]; 8] = [TransferLut::IDENTITY.0; 8];
        let id = TransferLut::IDENTITY.as_array();
        let inv = TransferLut::INVERTED.as_array();

        let pipe = PipeState {
            blend_mode: BlendMode::Normal,
            a_input: 255,
            overprint_mask: 0xFFFF_FFFF,
            overprint_additive: false,
            transfer: TransferSet {
                rgb: [inv; 3],
                gray: id,
                cmyk: [id; 4],
                device_n: &DN_ID,
            },
            soft_mask: None,
            alpha0: None,
            knockout: false,
            knockout_opacity: 255,
            non_isolated_group: false,
        };
        assert!(
            !pipe.transfer.is_identity_rgb(),
            "test prerequisite: inverting LUT must not register as identity"
        );

        let color = [200u8, 100, 50];
        let src = PipeSrc::Solid(&color);
        let shape = [255u8; 4]; // full coverage → general path writes src, then applies transfer
        let mut dst = vec![0u8; 12];

        render_span_aa::<Rgb8>(&pipe, &src, &mut dst, None, &shape, 0, 3, 0);

        // General path: full coverage → `apply_transfer_pixel` runs and
        // emits `255 - src`. Fast path would emit `src` unchanged.
        for px in 0..4 {
            assert_eq!(
                &dst[px * 3..px * 3 + 3],
                &[55, 155, 205],
                "pixel {px}: transfer LUT must invert each channel; \
                 if the fast-path gate mis-fired, dst would be [200, 100, 50]"
            );
        }
    }

    /// `TransferSet::is_identity_rgb()` gates the fast path; when it returns
    /// `true`, `composite_aa_rgb8_opaque` runs. If the predicate mis-reports
    /// `false` for a genuinely-identity LUT (cargo-mutants whole-body
    /// → `false` survives without this test), the general path runs and
    /// uses a higher-precision `div255` than the fast path, producing
    /// different output bytes on some inputs.
    ///
    /// Pin the byte values that the fast path produces on a representative
    /// large span; the general path's higher-precision `div255` would shift
    /// at least one byte by ≥ 1 LSB on this input set.
    #[test]
    fn identity_transfer_takes_fast_path_with_pinned_bytes() {
        let pipe = aa_pipe();
        assert!(
            pipe.transfer.is_identity_rgb(),
            "test prerequisite: aa_pipe() must register as identity"
        );

        // 17 pixels: crosses the LANE=16 boundary, exercising both the
        // chunked path and the scalar tail.
        let color = [200u8, 100, 50];
        let src = PipeSrc::Solid(&color);
        let shape: Vec<u8> = (0u8..17).map(|i| i.wrapping_mul(17)).collect();
        let initial: Vec<u8> = (0u8..51).map(|i| i.wrapping_mul(13)).collect();
        let mut dst_fast = initial.clone();

        render_span_aa::<Rgb8>(&pipe, &src, &mut dst_fast, None, &shape, 0, 16, 0);

        // Compute the reference via the fast path's formula:
        //   a_src   = (a_input * shape[i] + 255) >> 8
        //   c_out_j = ((255 - a_src) * c_dst[j] + a_src * src[j] + 255) >> 8
        let a_in = 255u16;
        let mut expected = initial;
        for (i, &sh) in shape.iter().enumerate() {
            let a_src = (a_in * u16::from(sh) + 255) >> 8;
            let inv = 255 - a_src;
            let b = i * 3;
            for (j, sc) in color.iter().enumerate() {
                let v = (inv * u16::from(expected[b + j]) + a_src * u16::from(*sc) + 255) >> 8;
                // `v` is bounded by the fast path's div255 (`(.. + 255) >> 8` ≤ 255).
                expected[b + j] = u8::try_from(v).expect("fast-path div255 result must fit in u8");
            }
        }
        assert_eq!(
            dst_fast, expected,
            "identity-LUT path must use the fast path's `(v + 255) >> 8` div255"
        );
    }

    /// The fast path (`composite_aa_rgb8_opaque` via `(v+255)>>8`) and the
    /// general no-alpha path (`render_span_aa_inner` via
    /// `color::convert::div255`'s exact `(v + v>>8 + 0x80) >> 8`) must agree
    /// to within 1 LSB per channel on every gate-eligible input.
    ///
    /// This pins the cross-path invariant that the deferred v0.9.2 audit
    /// (`audit/2026-05-11-avx512-fastpath-vs-general-byte-equality.md`)
    /// flagged: each path's *internal* byte values are pinned by sibling
    /// tests, but a regression that swapped one `div255` shape for the
    /// other — or that mis-classified an identity-LUT call as needing the
    /// general path, or vice-versa — could silently shift output by 1 LSB
    /// and only be caught by the pixel-diff integration suite against
    /// pdftoppm.
    ///
    /// Strategy:
    /// 1. Build a deterministic corpus of `(color, a_input, shape, dst)`
    ///    tuples that span the gate-firing domain (full opacity, partial,
    ///    zero; full/partial/zero coverage; saturated/mid-range/dark
    ///    destination bytes).
    /// 2. For each tuple, run the fast path via `render_span_aa::<Rgb8>`
    ///    with `dst_alpha = None` and an identity transfer (so the gate
    ///    fires).
    /// 3. Compute the corresponding general-path reference using
    ///    `color::convert::div255` directly — the same algebra as the
    ///    `None` arm of `render_span_aa_inner` at `aa.rs:192-207`, only
    ///    `apply_transfer_in_place` is a no-op under the identity LUT, so
    ///    it can be skipped.
    /// 4. Assert `|fast[i] - reference[i]| <= 1` for every byte.
    ///
    /// Why `<= 1` and not `==`: the two `div255` formulas round
    /// differently. The fast path's `(v + 255) >> 8` is correct
    /// to within ±1 LSB; the general path's `(v + v>>8 + 0x80) >> 8` is
    /// exact. The difference can be 1 LSB on specific `(c_dst, c_src,
    /// a_src)` triples (e.g. `inv=128, c_dst=255, a_src=127, c_src=255`:
    /// numerator 65025, fast=255, exact=255 — same; but
    /// `inv=128, c_dst=1, a_src=127, c_src=1`: num=255, fast=1, exact=1 —
    /// also same; the divergence shows up on numerators near 256k
    /// boundaries). What matters for downstream correctness is the
    /// ceiling, not exactness.
    ///
    /// Span length 23 = LANE(16) + 7: wider than the fast-path's LANE-chunked
    /// branch so the chunked + scalar-tail loops both fire.
    const ONE_LSB_CORPUS_SPAN: usize = 23;

    /// Run the gate-eligible (no alpha plane, identity transfer) fast path
    /// over `initial`, returning the result. The test's "ground truth" leg
    /// is `run_exact_reference`.
    fn run_fast_path(color: [u8; 3], a_input: u8, shape: &[u8], initial: &[u8]) -> Vec<u8> {
        let mut dst = initial.to_vec();
        let mut pipe = aa_pipe();
        pipe.a_input = a_input;
        let src = PipeSrc::Solid(color.as_slice());
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_possible_wrap,
            reason = "ONE_LSB_CORPUS_SPAN = 23 fits i32 trivially"
        )]
        let x1 = (ONE_LSB_CORPUS_SPAN - 1) as i32;
        render_span_aa::<Rgb8>(&pipe, &src, &mut dst, None, shape, 0, x1, 0);
        dst
    }

    /// Inline algebra of the general no-alpha path (`aa.rs:192-207`) under
    /// an identity transfer — `apply_transfer_in_place` is a no-op so the
    /// loop reduces to per-channel
    /// `div255((255 - a_src) * c_dst + a_src * c_src)` using the exact
    /// `color::convert::div255`.
    fn run_exact_reference(color: [u8; 3], a_input: u8, shape: &[u8], initial: &[u8]) -> Vec<u8> {
        use color::convert::div255 as exact_div255;
        let mut dst = initial.to_vec();
        let a_in_u32 = u32::from(a_input);
        for (i, &sh) in shape.iter().enumerate() {
            let a_src = u32::from(exact_div255(a_in_u32 * u32::from(sh)));
            let base = i * 3;
            for j in 0..3 {
                let c_dst = u32::from(dst[base + j]);
                let c_src = u32::from(color[j]);
                dst[base + j] = exact_div255((255 - a_src) * c_dst + a_src * c_src);
            }
        }
        dst
    }

    /// Per-byte `|fast - exact| ≤ 1` assertion. Returns the number of bytes
    /// that diverged by exactly 1 LSB (the trip-wire counter for the outer
    /// "corpus actually exercises the divergence" sanity check).
    fn assert_within_one_lsb(
        fast: &[u8],
        exact: &[u8],
        color: [u8; 3],
        a_input: u8,
        shape: &[u8],
        initial: &[u8],
    ) -> usize {
        let mut one_lsb_count = 0usize;
        for (i, (&f, &r)) in fast.iter().zip(exact.iter()).enumerate() {
            let diff = i32::from(f).abs_diff(i32::from(r));
            assert!(
                diff <= 1,
                "byte {i}: fast={f}, exact={r}, diff={diff}; \
                 colour={color:?}, a_input={a_input}, \
                 shape[{i_px}]={sh}, initial[{i}]={init}",
                i_px = i / 3,
                sh = shape[i / 3],
                init = initial[i],
            );
            if diff == 1 {
                one_lsb_count += 1;
            }
        }
        one_lsb_count
    }

    #[test]
    fn fast_path_matches_general_div255_within_one_lsb() {
        let pipe = aa_pipe();
        assert!(
            pipe.transfer.is_identity_rgb(),
            "test prerequisite: aa_pipe() must register as identity so the gate fires"
        );

        // Corpus: cross-product of representative source colours, alpha
        // inputs, shape patterns, and initial destinations.
        let colours: [[u8; 3]; 5] = [
            [0, 0, 0],       // black
            [255, 255, 255], // white
            [200, 100, 50],  // mid-saturated warm
            [1, 254, 127],   // off-by-one boundaries
            [128, 128, 128], // 50%-grey
        ];
        let a_inputs: [u8; 4] = [0, 1, 128, 255];
        // Shape patterns: full coverage, zero, ramp, alternating.
        let shape_patterns: [Box<dyn Fn(usize) -> u8>; 4] = [
            Box::new(|_| 255),
            Box::new(|_| 0),
            #[expect(
                clippy::cast_possible_truncation,
                reason = "i ∈ [0,23] * 11 ≤ 253 fits u8"
            )]
            Box::new(|i| (i * 11) as u8),
            Box::new(|i| if i % 2 == 0 { 255 } else { 64 }),
        ];
        // Initial destinations: black, white, ramp, alternating.
        let dst_patterns: [Box<dyn Fn(usize) -> u8>; 4] = [
            Box::new(|_| 0),
            Box::new(|_| 255),
            #[expect(
                clippy::cast_possible_truncation,
                reason = "i ∈ [0, COUNT*3) * 7 mod 256 fits u8"
            )]
            Box::new(|i| ((i * 7) % 256) as u8),
            Box::new(|i| if i % 2 == 0 { 0 } else { 200 }),
        ];

        let mut total_cases = 0usize;
        let mut total_bytes_at_one_lsb = 0usize;

        for &color in &colours {
            for &a_input in &a_inputs {
                for sh_fn in &shape_patterns {
                    for dst_fn in &dst_patterns {
                        let shape: Vec<u8> = (0..ONE_LSB_CORPUS_SPAN).map(&**sh_fn).collect();
                        let initial: Vec<u8> =
                            (0..ONE_LSB_CORPUS_SPAN * 3).map(&**dst_fn).collect();
                        let fast = run_fast_path(color, a_input, &shape, &initial);
                        let exact = run_exact_reference(color, a_input, &shape, &initial);
                        total_bytes_at_one_lsb +=
                            assert_within_one_lsb(&fast, &exact, color, a_input, &shape, &initial);
                        total_cases += 1;
                    }
                }
            }
        }

        // Sanity: the corpus should actually exercise the 1-LSB divergence
        // somewhere, otherwise the test is silently equivalent to byte
        // equality and the deferred audit's premise was wrong.  Empirically
        // this corpus produces ≥1 byte at diff=1; if it stops doing so, the
        // corpus needs widening before we trust the ≤1 LSB ceiling.
        assert!(
            total_bytes_at_one_lsb > 0,
            "corpus of {total_cases} cases produced no 1-LSB divergence — \
             corpus is too narrow to actually pin the ≤1 LSB ceiling"
        );
    }
}
