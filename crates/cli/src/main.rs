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

    // Resolve page range (convert to 1-based inclusive).
    let first = args.first_page.max(1);
    let last = args.last_page.unwrap_or(total).min(total);

    if first > last {
        eprintln!("pdf-raster: first page ({first}) is after last page ({last})");
        std::process::exit(1);
    }

    // Collect page numbers to render (filter odd/even if requested).
    let pages: Vec<i32> = (first..=last)
        .filter(|&p| {
            if args.odd_only && args.even_only {
                true
            }
            // both set → all pages
            else if args.odd_only {
                p % 2 == 1
            } else if args.even_only {
                p % 2 == 0
            } else {
                true
            }
        })
        .take(if args.single_file { 1 } else { usize::MAX })
        .collect();

    // Render pages in parallel.
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

    // Report errors.
    let had_errors = !errors.is_empty();
    for (page, err) in errors {
        eprintln!("pdf-raster: page {page}: {err}");
    }

    if had_errors {
        std::process::exit(1);
    }
}
