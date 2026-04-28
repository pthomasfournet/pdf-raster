mod args;
mod naming;
mod render;

use std::sync::atomic::{AtomicU32, Ordering};
#[cfg(any(feature = "gpu-aa", feature = "nvjpeg", feature = "gpu-icc"))]
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use rayon::prelude::*;

use args::Args;

fn main() {
    // try_init so tests and embedders that already called init() don't panic.
    let _ = env_logger::try_init();

    let args = Args::parse();

    // Initialise the GPU context once; each page render receives a clone of the
    // Arc so they can issue independent streams without re-init overhead.
    // A warning (not a hard error) is emitted when the feature is compiled in but
    // the GPU is unavailable — the CPU fallback path activates automatically.
    #[cfg(any(feature = "gpu-aa", feature = "nvjpeg", feature = "gpu-icc"))]
    let gpu_ctx: Option<Arc<gpu::GpuCtx>> = match gpu::GpuCtx::init() {
        Ok(ctx) => Some(Arc::new(ctx)),
        Err(e) => {
            eprintln!("pdf-raster: GPU unavailable ({e}); falling back to CPU");
            None
        }
    };

    // Build rayon thread pool.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.num_threads)
        .build()
        .expect("failed to build thread pool");

    if args.gray || args.mono {
        eprintln!(
            "pdf-raster: warning: --gray/--mono are not yet supported by the native \
             renderer and will be ignored"
        );
    }

    let doc = pdf_interp::open(&args.input).unwrap_or_else(|e| {
        eprintln!("pdf-raster: failed to open PDF: {e}");
        std::process::exit(1);
    });

    let n = pdf_interp::page_count(&doc);
    if n == 0 {
        eprintln!("pdf-raster: document has no pages");
        std::process::exit(1);
    }
    let total = i32::try_from(n).unwrap_or_else(|_| {
        eprintln!("pdf-raster: document has too many pages ({n} > i32::MAX)");
        std::process::exit(1);
    });

    let pages = build_page_list(total, &args);

    let n_pages = pages.len();
    let done = AtomicU32::new(0);
    let start = Instant::now();

    #[expect(
        clippy::cast_sign_loss,
        reason = "total validated ≥ 1 via i32::try_from above"
    )]
    let total_u32 = total as u32;

    let errors: Vec<(i32, render::RenderError)> = pool.install(|| {
        pages
            .par_iter()
            .filter_map(|&page_num| {
                #[expect(
                    clippy::cast_sign_loss,
                    reason = "page_num ≥ 1, enforced by build_page_list"
                )]
                let page_u32 = page_num as u32;
                #[cfg(any(feature = "gpu-aa", feature = "nvjpeg", feature = "gpu-icc"))]
                let result = render::render_page_native(&doc, page_u32, total_u32, &args, gpu_ctx.as_ref());
                #[cfg(not(any(feature = "gpu-aa", feature = "nvjpeg", feature = "gpu-icc")))]
                let result = render::render_page_native(&doc, page_u32, total_u32, &args);
                report_progress(&args, &done, n_pages, &start, page_num);
                result.err().map(|e| (page_num, e))
            })
            .collect()
    });

    report_errors(errors);
}

/// Build the filtered, clamped list of 1-based page numbers to render.
///
/// Exits the process with status 1 if the resulting range is empty (first > last
/// after clamping).  Boundary violations that can be clamped are warned and
/// clamped rather than rejected.
fn build_page_list(total: i32, args: &Args) -> Vec<i32> {
    let requested_first = args.first_page;
    let requested_last = args.last_page.unwrap_or(total);

    let first = requested_first.max(1);
    let last = requested_last.min(total);

    if requested_first < 1 {
        eprintln!("pdf-raster: warning: first page {requested_first} < 1; clamped to 1");
    }
    if requested_last > total {
        eprintln!(
            "pdf-raster: warning: last page {requested_last} exceeds document length ({total}); \
             clamped to {total}"
        );
    }

    if first > last {
        eprintln!(
            "pdf-raster: first page ({first}) is after last page ({last}); nothing to render"
        );
        std::process::exit(1);
    }

    (first..=last)
        .filter(|&p| match (args.odd_only, args.even_only) {
            (true, false) => p % 2 == 1,
            (false, true) => p % 2 == 0,
            // both set or neither set → no filtering
            (true, true) | (false, false) => true,
        })
        .take(if args.single_file { 1 } else { usize::MAX })
        .collect()
}

/// Print per-page progress to stderr if `--progress` was requested.
fn report_progress(args: &Args, done: &AtomicU32, n_pages: usize, start: &Instant, page_num: i32) {
    if !args.progress {
        return;
    }
    let completed = done.fetch_add(1, Ordering::Relaxed) + 1;
    let elapsed = start.elapsed().as_secs_f64();
    let rate = f64::from(completed) / elapsed;
    let remaining = n_pages.saturating_sub(completed as usize);
    if rate > 0.0 {
        #[expect(
            clippy::cast_precision_loss,
            reason = "ETA display; ±1s accuracy is sufficient"
        )]
        let eta_s = remaining as f64 / rate;
        eprintln!(
            "pdf-raster: page {page_num} done  [{completed}/{n_pages}]  \
             {elapsed:.1}s elapsed  ~{eta_s:.1}s remaining"
        );
    } else {
        eprintln!("pdf-raster: page {page_num} done  [{completed}/{n_pages}]");
    }
}

/// Print all render errors sorted by page number, then exit with status 1 if any.
fn report_errors(mut errors: Vec<(i32, render::RenderError)>) {
    if errors.is_empty() {
        return;
    }
    errors.sort_by_key(|(p, _)| *p);
    for (page, err) in &errors {
        eprintln!("pdf-raster: page {page}: {err}");
        let mut src = std::error::Error::source(err);
        while let Some(cause) = src {
            eprintln!("  caused by: {cause}");
            src = cause.source();
        }
    }
    std::process::exit(1);
}
