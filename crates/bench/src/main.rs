//! Synthetic fill-path benchmark comparing rasterizer throughput.
//!
//! Renders N star polygons into a 1024×1024 buffer and reports throughput.
//! Both engines get pre-built path objects; only rasterization is timed.
//!
//! Run: cargo run -p bench --release -- [--iters N] [--stars N]

use std::f64::consts::PI;
use std::time::Instant;

use vello_cpu::{
    Level, Pixmap, RenderContext, RenderMode, RenderSettings,
    kurbo::{BezPath, Point},
    peniko::Color,
};

use color::Rgb8;
use raster::{
    Bitmap, Clip, Path, PathBuilder, PipeSrc, PipeState, fill, state::TransferSet, types::BlendMode,
};

// ── Config ────────────────────────────────────────────────────────────────────

struct Config {
    stars: usize,
    iters: u32,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            stars: 200,
            iters: 30,
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
            "--stars" => {
                if let Some(v) = args.next() {
                    cfg.stars = v.parse().unwrap_or(cfg.stars);
                }
            }
            _ => {}
        }
    }
    cfg
}

const W: u32 = 1024;
const H: u32 = 1024;

// ── Shared path generation ────────────────────────────────────────────────────

#[expect(
    clippy::cast_precision_loss,
    reason = "star count fits in f64 mantissa for realistic values"
)]
fn star_params(stars: usize) -> Vec<(f64, f64, f64, f64, usize)> {
    let (w, h) = (f64::from(W), f64::from(H));
    (0..stars)
        .map(|i| {
            let t = i as f64 / stars as f64;
            let cx = w * (0.1 + 0.8 * ((t * 7.3).sin() * 0.5 + 0.5));
            let cy = h * (0.1 + 0.8 * ((t * 5.1).cos() * 0.5 + 0.5));
            let r = w.min(h) * (0.03 + 0.07 * ((t * 3.7).sin() * 0.5 + 0.5));
            (r, r * 0.4, cx, cy, 5 + i % 4)
        })
        .collect()
}

#[expect(
    clippy::cast_precision_loss,
    reason = "star vertex count fits in f64 mantissa"
)]
fn make_kurbo_star(cx: f64, cy: f64, r_o: f64, r_i: f64, n: usize) -> BezPath {
    let mut p = BezPath::new();
    for k in 0..n * 2 {
        let a = PI * k as f64 / n as f64 - PI / 2.0;
        let r = if k % 2 == 0 { r_o } else { r_i };
        let pt = Point::new(cx + r * a.cos(), cy + r * a.sin());
        if k == 0 {
            p.move_to(pt);
        } else {
            p.line_to(pt);
        }
    }
    p.close_path();
    p
}

#[expect(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    reason = "star geometry; usize vertex index fits in f64"
)]
fn make_raster_star(cx: f64, cy: f64, r_o: f64, r_i: f64, n: usize) -> Path {
    let mut b = PathBuilder::new();
    for k in 0..n * 2 {
        let a = PI * k as f64 / n as f64 - PI / 2.0;
        let r = if k % 2 == 0 { r_o } else { r_i };
        let (x, y) = (cx + r * a.cos(), cy + r * a.sin());
        if k == 0 {
            b.move_to(x, y).expect("move_to");
        } else {
            b.line_to(x, y).expect("line_to");
        }
    }
    b.close(true).expect("close");
    b.build()
}

// 8 distinct colours.
const VELLO_COLORS: [Color; 8] = [
    Color::from_rgba8(220, 64, 64, 255),
    Color::from_rgba8(64, 220, 64, 255),
    Color::from_rgba8(64, 64, 220, 255),
    Color::from_rgba8(64, 220, 220, 255),
    Color::from_rgba8(220, 64, 220, 255),
    Color::from_rgba8(220, 220, 64, 255),
    Color::from_rgba8(220, 128, 64, 255),
    Color::from_rgba8(128, 128, 128, 255),
];
const RASTER_COLORS: [[u8; 3]; 8] = [
    [220, 64, 64],
    [64, 220, 64],
    [64, 64, 220],
    [64, 220, 220],
    [220, 64, 220],
    [220, 220, 64],
    [220, 128, 64],
    [128, 128, 128],
];

// ── vello_cpu benchmark ───────────────────────────────────────────────────────

fn bench_vello(cfg: &Config, params: &[(f64, f64, f64, f64, usize)]) -> f64 {
    let paths: Vec<BezPath> = params
        .iter()
        .map(|&(r_o, r_i, cx, cy, n)| make_kurbo_star(cx, cy, r_o, r_i, n))
        .collect();

    // Single-threaded to match our raster benchmark.
    let settings = RenderSettings {
        level: Level::try_detect().unwrap_or(Level::baseline()),
        num_threads: 0,
        render_mode: RenderMode::OptimizeSpeed,
    };
    #[expect(
        clippy::cast_possible_truncation,
        reason = "W and H are compile-time constants ≤ 1024"
    )]
    let mut pixmap = Pixmap::new(W as u16, H as u16);

    // Warmup.
    {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "W and H are compile-time constants ≤ 1024"
        )]
        let mut ctx = RenderContext::new_with(W as u16, H as u16, settings);
        for (i, p) in paths.iter().enumerate() {
            ctx.set_paint(VELLO_COLORS[i % VELLO_COLORS.len()]);
            ctx.fill_path(p);
        }
        ctx.flush();
        ctx.render_to_pixmap(&mut pixmap);
    }

    let t0 = Instant::now();
    for _ in 0..cfg.iters {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "W and H are compile-time constants ≤ 1024"
        )]
        let mut ctx = RenderContext::new_with(W as u16, H as u16, settings);
        for (i, p) in paths.iter().enumerate() {
            ctx.set_paint(VELLO_COLORS[i % VELLO_COLORS.len()]);
            ctx.fill_path(p);
        }
        ctx.flush();
        ctx.render_to_pixmap(&mut pixmap);
    }
    t0.elapsed().as_secs_f64() * 1000.0 / f64::from(cfg.iters)
}

// ── our raster benchmark ──────────────────────────────────────────────────────

fn bench_ours(cfg: &Config, params: &[(f64, f64, f64, f64, usize)]) -> f64 {
    let identity = [1.0f64, 0.0, 0.0, 1.0, 0.0, 0.0];
    let transfer = TransferSet::identity_rgb();
    let pipe = PipeState {
        blend_mode: BlendMode::Normal,
        a_input: 255,
        overprint_mask: 0xFFFF_FFFF,
        overprint_additive: false,
        transfer,
        soft_mask: None,
        alpha0: None,
        knockout: false,
        knockout_opacity: 255,
        non_isolated_group: false,
    };
    let paths: Vec<Path> = params
        .iter()
        .map(|&(r_o, r_i, cx, cy, n)| make_raster_star(cx, cy, r_o, r_i, n))
        .collect();
    let clip = Clip::new(0.0, 0.0, f64::from(W), f64::from(H), false);

    // Warmup.
    {
        let mut bitmap = Bitmap::<Rgb8>::new(W, H, 1, false);
        for (i, path) in paths.iter().enumerate() {
            let color = RASTER_COLORS[i % RASTER_COLORS.len()];
            fill::<Rgb8>(
                &mut bitmap,
                &clip,
                path,
                &pipe,
                &PipeSrc::Solid(&color),
                &identity,
                1.0,
                true,
            );
        }
    }

    let t0 = Instant::now();
    for _ in 0..cfg.iters {
        let mut bitmap = Bitmap::<Rgb8>::new(W, H, 1, false);
        for (i, path) in paths.iter().enumerate() {
            let color = RASTER_COLORS[i % RASTER_COLORS.len()];
            fill::<Rgb8>(
                &mut bitmap,
                &clip,
                path,
                &pipe,
                &PipeSrc::Solid(&color),
                &identity,
                1.0,
                true,
            );
        }
    }
    t0.elapsed().as_secs_f64() * 1000.0 / f64::from(cfg.iters)
}

// ── main ─────────────────────────────────────────────────────────────────────

fn main() {
    let cfg = parse_config();
    println!(
        "Synthetic fill benchmark  {}×{}  {} stars  {} iters",
        W, H, cfg.stars, cfg.iters
    );
    let params = star_params(cfg.stars);
    let ms_v = bench_vello(&cfg, &params);
    let ms_o = bench_ours(&cfg, &params);
    println!("  vello_cpu : {ms_v:8.2} ms/frame");
    println!("  pdf-raster: {ms_o:8.2} ms/frame");
    if ms_o < ms_v {
        println!("  → pdf-raster is {:.2}× faster", ms_v / ms_o);
    } else {
        println!("  → vello_cpu  is {:.2}× faster", ms_o / ms_v);
    }
}
