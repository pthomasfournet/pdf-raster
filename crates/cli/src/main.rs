mod args;
mod diagnostics;
mod naming;
mod page_queue;
mod render;

use std::sync::atomic::AtomicU32;
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

    let tasks = pages.iter().map(|&page_num| page_queue::PageTask {
        page_num,
        hint: page_queue::RoutingHint::Unclassified,
    });
    let errors: Vec<(i32, render::RenderError)> = page_queue::PageQueue::new().run(
        tasks,
        &pool,
        &session,
        total_u32,
        &args,
        &page_queue::ProgressCtx {
            done: &done,
            n_pages,
            start: &start,
        },
    );

    // Eagerly drop GPU decoders on every worker thread while the CUDA driver is
    // still fully live, before the pool drops.  Avoids the process-exit teardown
    // race where all workers call nvjpegJpegStateDestroy concurrently into a
    // driver that has already started its own atexit shutdown sequence.
    let _ = pool.broadcast(|_| pdf_raster::release_gpu_decoders());

    diagnostics::report_errors(errors);
}
