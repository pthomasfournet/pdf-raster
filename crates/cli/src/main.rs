mod args;
mod diagnostics;
mod naming;
mod render;

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use args::Args;
use clap::Parser;
use rayon::prelude::*;

fn main() {
    let _ = env_logger::try_init();

    let args = Args::parse();

    if args.odd_only && args.even_only {
        eprintln!("pdf-raster: --odd and --even are mutually exclusive");
        std::process::exit(1);
    }
    if let Err(e) = args.validate_format_flags() {
        eprintln!("pdf-raster: {e}");
        std::process::exit(1);
    }

    let session_config = args.session_config().unwrap_or_else(|e| {
        eprintln!("pdf-raster: {e}");
        std::process::exit(1);
    });

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.num_threads)
        .thread_name(|i| format!("raster-worker-{i}"))
        // 8 MiB per worker: path flattening and Bézier subdivision recurse deeply
        // on degenerate inputs and overflow the 2 MiB rayon default.
        .stack_size(8 * 1024 * 1024)
        .build()
        .expect("failed to build thread pool");

    let session = pdf_raster::open_session(std::path::Path::new(&args.input), &session_config)
        .unwrap_or_else(|e| {
            diagnostics::report_open_error(&e, &args);
            std::process::exit(1);
        });

    let n = session.total_pages();
    if n == 0 {
        eprintln!("pdf-raster: document has no pages");
        std::process::exit(1);
    }
    let total = i32::try_from(n).unwrap_or_else(|_| {
        eprintln!("pdf-raster: document has too many pages ({n} > i32::MAX)");
        std::process::exit(1);
    });

    let (pages, page_warnings) = args.build_page_list(total).unwrap_or_else(|e| {
        eprintln!("pdf-raster: {e}");
        std::process::exit(1);
    });
    for w in &page_warnings {
        eprintln!("pdf-raster: warning: {w}");
    }

    let n_pages = pages.len();
    let done = AtomicU32::new(0);
    let start = Instant::now();

    #[expect(
        clippy::cast_sign_loss,
        reason = "total ≥ 1 guaranteed by the n == 0 exit above; i32::try_from only guards the upper bound"
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

                let result = render::render_page(&session, page_u32, total_u32, &args);

                if args.progress {
                    let completed = done.fetch_add(1, Ordering::Relaxed) + 1;
                    let elapsed = self::elapsed_secs(&start);
                    let completed_usize = usize::try_from(completed).unwrap_or(n_pages);
                    let remaining = n_pages.saturating_sub(completed_usize);
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "ETA display; ±1s accuracy is sufficient"
                    )]
                    let eta_str = if elapsed > 0.5 && completed >= 2 {
                        let rate = f64::from(completed) / elapsed;
                        let eta_s = remaining as f64 / rate;
                        if eta_s.is_finite() {
                            format!("~{eta_s:.1}s remaining")
                        } else {
                            "~?s remaining".to_owned()
                        }
                    } else {
                        "~?s remaining".to_owned()
                    };
                    eprintln!(
                        "pdf-raster: page {page_num} done  [{completed}/{n_pages}]  \
                         {elapsed:.1}s elapsed  {eta_str}"
                    );
                }

                result.err().map(|e| (page_num, e))
            })
            .collect()
    });

    // Eagerly drop GPU decoders on every worker thread while the CUDA driver is
    // still fully live, before the pool drops.  Avoids the process-exit teardown
    // race where all workers call nvjpegJpegStateDestroy concurrently into a
    // driver that has already started its own atexit shutdown sequence.
    let _ = pool.broadcast(|_| pdf_raster::release_gpu_decoders());

    diagnostics::report_errors(errors);
}

#[inline]
fn elapsed_secs(start: &Instant) -> f64 {
    start.elapsed().as_secs_f64()
}
