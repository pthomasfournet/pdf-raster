//! GPU dispatch threshold calibration.
// All timing values are display-only; f64 precision loss on u64 nanosecond
// counts (ratios, formatted durations) is intentional and inconsequential.
#![expect(
    clippy::cast_precision_loss,
    reason = "display-only timing ratios and ns→µs/ms conversions"
)]
//!
//! Measures the breakeven pixel count where GPU dispatch becomes faster than
//! the CPU fallback for each kernel, accounting for actual `PCIe` 5.0 H2D/D2H
//! latency on this machine.
//!
//! Run:
//! ```text
//! CUDA_ARCH=sm_120 cargo build --release -p gpu --bin threshold_bench
//! target/release/threshold_bench [--iters N] [--warmup N]
//! ```
//!
//! Output: a table per kernel and a recommended threshold per constant.

use std::time::Instant;

use rasterrocket_gpu::{GpuCtx, aa_fill_cpu, build_tile_records, icc_cmyk_to_rgb_cpu};

/// Unwrap a GPU `Result`, printing the error with context and exiting with
/// status 1.  Used inside timing closures where `?` is not available.
fn gpu_unwrap<T>(res: Result<T, Box<dyn std::error::Error>>, ctx: &str) -> T {
    res.unwrap_or_else(|e| {
        eprintln!("threshold_bench: GPU error in {ctx}: {e}");
        std::process::exit(1);
    })
}

// ── Config ─────────────────────────────────────────────────────────────────

struct Config {
    /// Number of timed iterations per (kernel, size) cell.
    iters: u32,
    /// Number of warmup iterations (timed but discarded).
    warmup: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            iters: 20,
            warmup: 5,
        }
    }
}

fn parse_config() -> Config {
    let mut cfg = Config::default();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--iters" => {
                if let Some(v) = args.next() {
                    cfg.iters = v.parse().unwrap_or(cfg.iters);
                }
            }
            "--warmup" => {
                if let Some(v) = args.next() {
                    cfg.warmup = v.parse().unwrap_or(cfg.warmup);
                }
            }
            _ => {}
        }
    }
    cfg
}

// ── Geometric pixel-count sweep ─────────────────────────────────────────────

/// Sizes to sweep: powers-of-2 from 2^8 to 2^22 (256 to 4M pixels).
fn sweep_sizes() -> Vec<usize> {
    (8u32..=22).map(|e| 1usize << e).collect()
}

/// Width/height for a near-square bounding box of `n` pixels.
fn dims(n: usize) -> (u32, u32) {
    // n ≤ 2^22 = 4M; sqrt ≤ 2048; fits u32 without truncation.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "n ≤ 4_194_304 (22-bit); exact in f64 (52-bit mantissa); sqrt ≤ 2049 — fits u32"
    )]
    let w = (n as f64).sqrt().ceil() as u32;
    #[expect(clippy::cast_possible_truncation, reason = "n ≤ 4_194_304 — fits u32")]
    let h = (n as u32).div_ceil(w);
    (w, h)
}

// ── Timing helpers ──────────────────────────────────────────────────────────

fn median_ns(mut samples: Vec<u64>) -> u64 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

/// Run `f` `warmup` times then `iters` times; return median duration in ns.
fn time_ns<F: FnMut()>(mut f: F, warmup: u32, iters: u32) -> u64 {
    for _ in 0..warmup {
        f();
    }
    // as_nanos() → u128; truncation to u64 is safe: u64::MAX ns ≈ 584 years.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "u64::MAX ns ≈ 584 years; no benchmark runs that long"
    )]
    let samples: Vec<u64> = (0..iters)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_nanos() as u64
        })
        .collect();
    median_ns(samples)
}

// ── Segment fixture: a square with `n_pixels` bbox ─────────────────────────

/// Return `[x0,y0,x1,y1, ...]` segments for the outline of a w×h rectangle
/// starting at (0,0), representing the 4 edges in CCW order.
fn square_segs(w: u32, h: u32) -> Vec<f32> {
    // w/h ≤ 2049 px; f32 has 23-bit mantissa (exact for integers ≤ 2^24).
    #[expect(
        clippy::cast_precision_loss,
        reason = "w/h ≤ 2049; exact in f32 (mantissa covers integers to 2^24)"
    )]
    let (w, h) = (w as f32, h as f32);
    // bottom, right, top, left — 4 segments
    vec![
        0.0, 0.0, w, 0.0, w, 0.0, w, h, w, h, 0.0, h, 0.0, h, 0.0, 0.0,
    ]
}

// ── AA fill benchmark ───────────────────────────────────────────────────────

fn bench_aa_fill(ctx: &GpuCtx, cfg: &Config) {
    println!("\n══ aa_fill: GPU warp-ballot vs CPU 64-sample ══");
    println!(
        "{:>10}  {:>12}  {:>12}  {:>8}  winner",
        "pixels", "cpu_ns", "gpu_ns", "ratio"
    );
    println!("{}", "─".repeat(60));

    let mut crossover: Option<usize> = None;

    for &n in &sweep_sizes() {
        let (w, h) = dims(n);
        let segs = square_segs(w, h);

        let cpu_ns = time_ns(
            || {
                let _ = aa_fill_cpu(&segs, 0.0, 0.0, w, h, false);
            },
            cfg.warmup,
            cfg.iters,
        );

        let gpu_ns = time_ns(
            || {
                let _ = gpu_unwrap(ctx.aa_fill_gpu(&segs, 0.0, 0.0, w, h, false), "aa_fill_gpu");
            },
            cfg.warmup,
            cfg.iters,
        );

        let ratio = cpu_ns as f64 / gpu_ns as f64;
        let winner = if gpu_ns < cpu_ns { "GPU ✓" } else { "CPU" };

        println!(
            "{:>10}  {:>12}  {:>12}  {:>7.2}x  {}",
            n,
            fmt_ns(cpu_ns),
            fmt_ns(gpu_ns),
            ratio,
            winner
        );

        if crossover.is_none() && gpu_ns < cpu_ns {
            crossover = Some(n);
        }
    }

    println!();
    match crossover {
        Some(px) => println!(
            "  → Recommended GPU_AA_FILL_THRESHOLD: {px} px  (current: {})",
            rasterrocket_gpu::GPU_AA_FILL_THRESHOLD
        ),
        None => println!("  → GPU never faster in this sweep — keep CPU or raise threshold"),
    }
}

// ── Tile fill benchmark ─────────────────────────────────────────────────────

fn bench_tile_fill(ctx: &GpuCtx, cfg: &Config) {
    println!("\n══ tile_fill: GPU analytical vs aa_fill_cpu ══");
    println!(
        "{:>10}  {:>12}  {:>12}  {:>8}  winner",
        "pixels", "aa_cpu_ns", "tile_gpu_ns", "ratio"
    );
    println!("{}", "─".repeat(60));

    let mut crossover: Option<usize> = None;

    for &n in &sweep_sizes() {
        let (w, h) = dims(n);
        let segs = square_segs(w, h);

        // CPU baseline: aa_fill_cpu (what the dispatch falls back to)
        let cpu_ns = time_ns(
            || {
                let _ = aa_fill_cpu(&segs, 0.0, 0.0, w, h, false);
            },
            cfg.warmup,
            cfg.iters,
        );

        // GPU: build tile records (CPU sort) + tile_fill kernel
        let gpu_ns = time_ns(
            || {
                let (records, starts, counts, grid_w) = build_tile_records(&segs, 0.0, 0.0, w, h);
                let _ = gpu_unwrap(
                    ctx.tile_fill(&records, &starts, &counts, grid_w, w, h, false),
                    "tile_fill",
                );
            },
            cfg.warmup,
            cfg.iters,
        );

        let ratio = cpu_ns as f64 / gpu_ns as f64;
        let winner = if gpu_ns < cpu_ns { "GPU ✓" } else { "CPU" };

        println!(
            "{:>10}  {:>12}  {:>12}  {:>7.2}x  {}",
            n,
            fmt_ns(cpu_ns),
            fmt_ns(gpu_ns),
            ratio,
            winner
        );

        if crossover.is_none() && gpu_ns < cpu_ns {
            crossover = Some(n);
        }
    }

    println!();
    match crossover {
        Some(px) => println!(
            "  → Recommended GPU_TILE_FILL_THRESHOLD: {px} px  (current: {})",
            rasterrocket_gpu::GPU_TILE_FILL_THRESHOLD
        ),
        None => println!("  → GPU never faster in this sweep — keep CPU or raise threshold"),
    }
}

// ── ICC CMYK→RGB benchmark ──────────────────────────────────────────────────

fn bench_icc_cmyk(ctx: &GpuCtx, cfg: &Config) {
    // NOTE: this bench measures the raw GPU matrix kernel (clut=None) against
    // CPU AVX-512.  The dispatch in icc_cmyk_to_rgb() always short-circuits to CPU
    // for clut=None regardless of pixel count, so the crossover below is informational
    // only.  To calibrate GPU_ICC_CLUT_THRESHOLD, re-run with a real CLUT workload.
    println!("\n══ icc_cmyk: GPU matrix kernel vs CPU AVX-512 (matrix path, informational) ══");
    println!(
        "{:>10}  {:>12}  {:>12}  {:>8}  winner",
        "pixels", "cpu_ns", "gpu_ns", "ratio"
    );
    println!("{}", "─".repeat(60));

    let mut crossover: Option<usize> = None;

    for &n in &sweep_sizes() {
        // CMYK input: alternating test values covering full range
        // i % 256 is in 0..=255 — truncation to u8 is exact by construction.
        #[expect(clippy::cast_possible_truncation, reason = "i % 256 is in 0..=255")]
        let cmyk: Vec<u8> = (0..n * 4).map(|i| (i % 256) as u8).collect();

        let cpu_ns = time_ns(
            || {
                let _ = icc_cmyk_to_rgb_cpu(&cmyk, None);
            },
            cfg.warmup,
            cfg.iters,
        );

        // Use the unconditional GPU path to measure actual dispatch cost at
        // all sizes, not just those above the current threshold.
        let gpu_ns = time_ns(
            || {
                let _ = gpu_unwrap(ctx.icc_cmyk_to_rgb_gpu(&cmyk, None), "icc_cmyk_to_rgb_gpu");
            },
            cfg.warmup,
            cfg.iters,
        );

        let ratio = cpu_ns as f64 / gpu_ns as f64;
        let winner = if gpu_ns < cpu_ns { "GPU ✓" } else { "CPU" };

        println!(
            "{:>10}  {:>12}  {:>12}  {:>7.2}x  {}",
            n,
            fmt_ns(cpu_ns),
            fmt_ns(gpu_ns),
            ratio,
            winner
        );

        if crossover.is_none() && gpu_ns < cpu_ns {
            crossover = Some(n);
        }
    }

    println!();
    match crossover {
        Some(px) => println!(
            "  → Matrix kernel crossover at {px} px — but dispatch always uses CPU for \
             clut=None; GPU_ICC_CLUT_THRESHOLD={} applies to CLUT path only",
            rasterrocket_gpu::GPU_ICC_CLUT_THRESHOLD,
        ),
        None => println!(
            "  → GPU matrix kernel never faster — CPU AVX-512 dominates at all sizes. \
             Dispatch short-circuits to CPU for clut=None (expected)."
        ),
    }
}

// ── Formatting ──────────────────────────────────────────────────────────────

fn fmt_ns(ns: u64) -> String {
    if ns >= 1_000_000 {
        format!("{:.2} ms", ns as f64 / 1_000_000.0)
    } else if ns >= 1_000 {
        format!("{:.1} µs", ns as f64 / 1_000.0)
    } else {
        format!("{ns} ns")
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let cfg = parse_config();

    println!("GPU dispatch threshold calibration");
    println!("iters={}, warmup={}", cfg.iters, cfg.warmup);
    println!("Machine: Ryzen 9 9900X3D + RTX 5070 (CC 12.0, PCIe 5.0)");

    let ctx = GpuCtx::init().unwrap_or_else(|e| {
        eprintln!("threshold_bench: GPU init failed: {e}");
        eprintln!("Run `nvidia-smi` to verify the driver is loaded.");
        std::process::exit(1);
    });

    bench_aa_fill(&ctx, &cfg);
    bench_tile_fill(&ctx, &cfg);
    bench_icc_cmyk(&ctx, &cfg);

    println!("\nDone.");
}
