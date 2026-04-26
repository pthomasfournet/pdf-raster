mod args;
mod naming;
mod render;

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use clap::Parser;
use rayon::prelude::*;

use args::Args;
use pdf_bridge::{Document as PopplerDoc, install_log_callback};

fn main() {
    // Silence poppler's stderr before opening any document.  Must come first.
    install_log_callback();
    // try_init so tests and embedders that already called init() don't panic.
    let _ = env_logger::try_init();

    let args = Args::parse();

    // Build rayon thread pool.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.num_threads)
        .build()
        .expect("failed to build thread pool");

    // ── Document opening ──────────────────────────────────────────────────────

    // For the native path, open with lopdf.  For the poppler path, open with
    // pdf_bridge.  We always silence poppler callbacks regardless of the path
    // because the install_log_callback call above is harmless.

    let total: i32;
    let pages: Vec<i32>;

    if args.native {
        let doc = pdf_interp::open(&args.input).unwrap_or_else(|e| {
            eprintln!("pdf-raster: failed to open PDF: {e}");
            std::process::exit(1);
        });

        let n = pdf_interp::page_count(&doc);
        if n == 0 {
            eprintln!("pdf-raster: document has no pages");
            std::process::exit(1);
        }
        #[expect(
            clippy::cast_possible_wrap,
            reason = "n ≤ u32::MAX; real documents have far fewer pages than i32::MAX"
        )]
        let n_i32 = n as i32;
        total = n_i32;

        pages = build_page_list(total, &args);
        if pages.is_empty() {
            eprintln!("pdf-raster: no pages match the requested range and filter");
            std::process::exit(1);
        }

        let n_pages = pages.len();
        let done = AtomicU32::new(0);
        let start = Instant::now();

        // total ≥ 1 (checked above) and ≤ u32::MAX (pdf_interp::page_count returns u32).
        #[expect(
            clippy::cast_sign_loss,
            reason = "total validated ≥ 1 and ≤ i32::MAX above; cast to u32 is safe"
        )]
        let total_u32 = total as u32;

        let errors: Vec<(i32, render::RenderError)> = pool.install(|| {
            pages
                .par_iter()
                .filter_map(|&page_num| {
                    // page_num ≥ 1 (enforced by build_page_list).
                    #[expect(
                        clippy::cast_sign_loss,
                        reason = "page_num ≥ 1; safe to cast to u32"
                    )]
                    let page_u32 = page_num as u32;
                    let result =
                        render::render_page_native(&doc, page_u32, total_u32, &args);
                    report_progress(&args, &done, n_pages, &start, page_num);
                    result.err().map(|e| (page_num, e))
                })
                .collect()
        });

        report_errors_and_exit(errors);
    } else {
        let doc = PopplerDoc::from_file(
            &args.input,
            args.owner_password.as_deref(),
            args.user_password.as_deref(),
        )
        .unwrap_or_else(|e| {
            eprintln!("pdf-raster: {e}");
            std::process::exit(1);
        });

        total = doc.page_count();
        if total <= 0 {
            eprintln!("pdf-raster: document has no pages");
            std::process::exit(1);
        }

        pages = build_page_list(total, &args);
        if pages.is_empty() {
            eprintln!("pdf-raster: no pages match the requested range and filter");
            std::process::exit(1);
        }

        let n_pages = pages.len();
        let done = AtomicU32::new(0);
        let start = Instant::now();

        let errors: Vec<(i32, render::RenderError)> = pool.install(|| {
            pages
                .par_iter()
                .filter_map(|&page_num| {
                    let result =
                        render::render_page_poppler(&doc, page_num, total, &args);
                    report_progress(&args, &done, n_pages, &start, page_num);
                    result.err().map(|e| (page_num, e))
                })
                .collect()
        });

        report_errors_and_exit(errors);
    }
}

/// Build the filtered, clamped list of 1-based page numbers to render.
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
            "pdf-raster: warning: last page {requested_last} exceeds document length ({total}); clamped to {total}"
        );
    }

    if first > last {
        eprintln!("pdf-raster: first page ({first}) is after last page ({last})");
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
fn report_progress(
    args: &Args,
    done: &AtomicU32,
    n_pages: usize,
    start: &Instant,
    page_num: i32,
) {
    if !args.progress {
        return;
    }
    let completed = done.fetch_add(1, Ordering::Relaxed) + 1;
    let elapsed = start.elapsed().as_secs_f64();
    let rate = f64::from(completed) / elapsed;
    let remaining = n_pages - completed as usize;
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

/// Print all render errors sorted by page number, then exit with status 1.
fn report_errors_and_exit(mut errors: Vec<(i32, render::RenderError)>) {
    errors.sort_by_key(|(p, _)| *p);
    for (page, err) in &errors {
        eprintln!("pdf-raster: page {page}: {err}");
        let mut src = std::error::Error::source(err);
        while let Some(cause) = src {
            eprintln!("  caused by: {cause}");
            src = cause.source();
        }
    }
    if !errors.is_empty() {
        std::process::exit(1);
    }
}
