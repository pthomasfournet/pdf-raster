mod args;
mod naming;
mod render;

use clap::Parser;
use rayon::prelude::*;

use args::Args;
use pdf_bridge::Document;

fn main() {
    let args = Args::parse();

    // Build rayon thread pool.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.num_threads)
        .build()
        .expect("failed to build thread pool");

    // Open the PDF document.
    let doc = Document::from_file(
        &args.input,
        args.owner_password.as_deref(),
        args.user_password.as_deref(),
    )
    .unwrap_or_else(|e| {
        eprintln!("pdf-raster: {e}");
        std::process::exit(1);
    });

    let total = doc.page_count();
    if total <= 0 {
        eprintln!("pdf-raster: document has no pages");
        std::process::exit(1);
    }

    // Resolve page range (1-based inclusive).
    // Clamp silently would hide user mistakes, so warn when the requested
    // range exceeds the document.
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

    // Collect page numbers to render, applying odd/even filter.
    let pages: Vec<i32> = (first..=last)
        .filter(|&p| match (args.odd_only, args.even_only) {
            (true, false) => p % 2 == 1,
            (false, true) => p % 2 == 0,
            // both set or neither set → no filtering
            (true, true) | (false, false) => true,
        })
        .take(if args.single_file { 1 } else { usize::MAX })
        .collect();

    if pages.is_empty() {
        eprintln!("pdf-raster: no pages match the requested range and filter");
        std::process::exit(1);
    }

    // Render pages in parallel and collect failures.
    // Errors are printed immediately by the collecting thread after the
    // parallel section so output ordering is deterministic.
    let errors: Vec<(i32, render::RenderError)> = pool.install(|| {
        pages
            .par_iter()
            .filter_map(|&page_num| {
                render::render_page(&doc, page_num, total, &args)
                    .err()
                    .map(|e| (page_num, e))
            })
            .collect()
    });

    // Report all failures; sort by page number for a predictable order.
    let mut errors = errors;
    errors.sort_by_key(|(p, _)| *p);
    for (page, err) in &errors {
        eprintln!("pdf-raster: page {page}: {err}");
        // Walk the error chain to surface the root cause.
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
