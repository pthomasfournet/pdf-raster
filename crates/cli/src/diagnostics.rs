//! Human-readable error reporting for the CLI.
//!
//! All stderr output that is not progress reporting lives here.
//! Functions are free-standing so `main` can call them without
//! owning any render state.

use pdf_raster::RasterError;

use crate::args::{Args, BackendArg};
use crate::render::RenderError;

/// Print the `source()` chain of `e` to stderr, one line per level.
pub fn print_error_chain(e: &dyn std::error::Error) {
    let mut src = e.source();
    while let Some(cause) = src {
        eprintln!("  caused by: {cause}");
        src = cause.source();
    }
}

/// Print a human-readable error (and actionable hints) when `open_session` fails.
pub fn report_open_error(e: &RasterError, args: &Args) {
    if matches!(e, RasterError::BackendUnavailable(_)) {
        eprintln!("pdf-raster: {e}");
        print_backend_hint(args);
    } else {
        eprintln!("pdf-raster: failed to open PDF: {e}");
        print_error_chain(e);
    }
}

/// Print a backend-specific hint after a `BackendUnavailable` error.
pub fn print_backend_hint(args: &Args) {
    match args.backend {
        BackendArg::Cuda => {
            eprintln!("  hint: --backend cuda requires a working CUDA driver and GPU.");
            eprintln!("        Run `nvidia-smi` to verify the driver is loaded.");
            eprintln!(
                "        Use --backend auto to fall back to CPU silently, or \
                 --backend cpu to skip GPU entirely."
            );
        }
        BackendArg::Vaapi => {
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

/// Sort errors by page number, print each with its cause chain, then exit 1.
///
/// No-op if `errors` is empty.
pub fn report_errors(mut errors: Vec<(i32, RenderError)>) {
    if errors.is_empty() {
        return;
    }
    errors.sort_by_key(|(p, _)| *p);
    for (page, err) in &errors {
        eprintln!("pdf-raster: page {page}: {err}");
        print_error_chain(err);
    }
    std::process::exit(1);
}
