mod args;
mod naming;
mod page_queue;
mod render;

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use args::Args;
use clap::Parser;

fn main() {
    let _ = env_logger::try_init();

    let args = Args::parse();

    // Validate mutually exclusive flags before allocating any resources.
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
            report_open_error(&e, &args);
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

    let pages = build_page_list(total, &args).unwrap_or_else(|| {
        // build_page_list already printed the reason.
        std::process::exit(1);
    });
    let n_pages = pages.len();
    let done = AtomicU32::new(0);
    let start = Instant::now();

    #[expect(
        clippy::cast_sign_loss,
        reason = "total validated ≥ 1 via i32::try_from above"
    )]
    let total_u32 = total as u32;

    let tasks = pages.iter().map(|&page_num| page_queue::PageTask {
        page_num,
        hint: page_queue::RoutingHint::Unclassified,
    });
    // Capacity = 2× thread count keeps workers fed while bounding peak
    // in-flight bitmap memory (vs par_iter which can start all N pages at once).
    let queue_capacity = args.num_threads.max(1) * 2;
    let errors: Vec<(i32, render::RenderError)> = pool.install(|| {
        page_queue::PageQueue::new(queue_capacity).run(
            tasks,
            &session,
            total_u32,
            &args,
            &page_queue::ProgressCtx {
                done: &done,
                n_pages,
                start: &start,
            },
        )
    });

    // Eagerly drop GPU decoders on every worker thread while the CUDA driver is
    // still fully live, before the pool drops.  Avoids the process-exit teardown
    // race where all workers call nvjpegJpegStateDestroy concurrently into a
    // driver that has already started its own atexit shutdown sequence.
    let _ = pool.broadcast(|_| pdf_raster::release_gpu_decoders());

    report_errors(errors);
}

/// Build the filtered, clamped list of 1-based page numbers to render.
///
/// Returns `None` (after printing a message to stderr) if no pages fall within
/// the requested range.  Never calls `std::process::exit`; callers decide how
/// to handle `None`.
fn build_page_list(total: i32, args: &Args) -> Option<Vec<i32>> {
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
        return None;
    }

    let pages: Vec<i32> = (first..=last)
        .filter(|&p| {
            // odd_only and even_only are mutually exclusive; main() enforces this
            // before build_page_list is called, so (true, true) is unreachable.
            if args.odd_only {
                p % 2 == 1
            } else if args.even_only {
                p % 2 == 0
            } else {
                true
            }
        })
        .take(if args.single_file { 1 } else { usize::MAX })
        .collect();

    if pages.is_empty() {
        eprintln!("pdf-raster: no pages selected by the current filter combination");
        return None;
    }

    Some(pages)
}

pub(crate) fn report_progress(
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
    // elapsed > 0 always: Instant::now() is taken before the pool starts work,
    // and progress is only reported after at least one page completes.
    let rate = f64::from(completed) / elapsed;
    // completed is a page counter; u32→usize is lossless on any 32-bit-or-wider target.
    let completed_usize = usize::try_from(completed).unwrap_or(n_pages);
    let remaining = n_pages.saturating_sub(completed_usize);
    #[expect(
        clippy::cast_precision_loss,
        reason = "ETA display; ±1s accuracy is sufficient"
    )]
    let eta_s = remaining as f64 / rate;
    eprintln!(
        "pdf-raster: page {page_num} done  [{completed}/{n_pages}]  \
         {elapsed:.1}s elapsed  ~{eta_s:.1}s remaining"
    );
}

/// Print a human-readable error (and actionable hints) when `open_session` fails.
fn report_open_error(e: &pdf_raster::RasterError, args: &Args) {
    if matches!(e, pdf_raster::RasterError::BackendUnavailable(_)) {
        eprintln!("pdf-raster: {e}");
        print_backend_hint(args);
    } else {
        eprintln!("pdf-raster: failed to open PDF: {e}");
        let mut src = std::error::Error::source(e);
        while let Some(cause) = src {
            eprintln!("  caused by: {cause}");
            src = cause.source();
        }
    }
}

/// Print a backend-specific hint after a `BackendUnavailable` error.
fn print_backend_hint(args: &Args) {
    match args.backend {
        args::BackendArg::Cuda => {
            eprintln!("  hint: --backend cuda requires a working CUDA driver and GPU.");
            eprintln!("        Run `nvidia-smi` to verify the driver is loaded.");
            eprintln!(
                "        Use --backend auto to fall back to CPU silently, or \
                 --backend cpu to skip GPU entirely."
            );
        }
        args::BackendArg::Vaapi => {
            eprintln!(
                "  hint: --backend vaapi could not open the DRM device ({}).",
                args.vaapi_device
            );
            eprintln!("        Verify the device exists and is readable by your user.");
            eprintln!("        Use --vaapi-device PATH to specify an alternate render node.");
            eprintln!(
                "        Use --backend auto to fall back to CPU silently, or \
                 --backend cpu to skip GPU entirely."
            );
        }
        _ => {
            eprintln!("  hint: use --backend auto to fall back to CPU when GPU is unavailable,");
            eprintln!("        or --backend cpu to force CPU-only mode.");
        }
    }
}

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
