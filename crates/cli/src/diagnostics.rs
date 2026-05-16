//! Human-readable error reporting for the CLI.
//!
//! All stderr output that is not progress reporting lives here.
//! Functions are free-standing so `main` can call them without
//! owning any render state.

use pdf_raster::{BackendPolicy, RasterError};

use crate::args::Args;
use crate::render::RenderError;

/// Print the `source()` chain of `e` to stderr, one line per level.
///
/// A level whose `Display` is identical to the text already printed is a
/// transparent wrapper (e.g. `RasterError::Pdf` delegating to its
/// `InterpError` source) and is skipped — repeating the same sentence under
/// "caused by:" is noise, not a diagnosis.
pub fn print_error_chain(e: &dyn std::error::Error) {
    let mut prev = e.to_string();
    let mut src = e.source();
    while let Some(cause) = src {
        let text = cause.to_string();
        if text != prev {
            eprintln!("  caused by: {text}");
            prev = text;
        }
        src = cause.source();
    }
}

/// Print a human-readable error (and actionable hints) when `open_session` fails.
///
/// `policy` is the resolved backend (after CLI flag → env var → default
/// fallback) so a hint can name the source the user actually controls
/// even when no `--backend` flag was passed.
pub fn report_open_error(e: &RasterError, args: &Args, policy: BackendPolicy) {
    if matches!(e, RasterError::BackendUnavailable(_)) {
        eprintln!("rrocket: {e}");
        print_backend_hint(policy, &args.vaapi_device, args.backend.is_none());
    } else {
        eprintln!("rrocket: failed to open PDF: {e}");
        print_error_chain(e);
    }
}

/// Print a backend-specific hint for the resolved policy.
///
/// `via_env` flags whether the policy came from `PDF_RASTER_BACKEND`
/// rather than `--backend`, so the hint can name the actual knob the
/// user reached for.
fn print_backend_hint(policy: BackendPolicy, vaapi_device: &str, via_env: bool) {
    #[cfg(not(feature = "vaapi"))]
    let _ = vaapi_device;
    let source_hint = |backend: &str| {
        if via_env {
            eprintln!("  hint: PDF_RASTER_BACKEND={backend} is set in your environment.");
        }
    };
    match policy {
        BackendPolicy::ForceCuda => {
            source_hint("cuda");
            print_cuda_hint();
        }
        #[cfg(feature = "vaapi")]
        BackendPolicy::ForceVaapi => {
            source_hint("vaapi");
            print_vaapi_hint(vaapi_device);
        }
        BackendPolicy::ForceVulkan => {
            source_hint("vulkan");
            print_vulkan_hint();
        }
        BackendPolicy::Auto | BackendPolicy::CpuOnly => {
            eprintln!("  hint: use --backend auto to fall back to CPU when GPU is unavailable,");
            eprintln!("        or --backend cpu to force CPU-only mode.");
        }
    }
}

fn print_cuda_hint() {
    eprintln!("        --backend cuda requires a working CUDA driver and GPU.");
    eprintln!("        Run `nvidia-smi` to verify the driver is loaded.");
    eprintln!(
        "        Use --backend auto to fall back to Vulkan / CPU silently, \
         or --backend cpu to skip GPU entirely."
    );
}

#[cfg(feature = "vaapi")]
fn print_vaapi_hint(device: &str) {
    eprintln!("        --backend vaapi could not open the DRM device ({device}).");
    eprintln!("        Verify the device exists and is readable by your user.");
    eprintln!("        Use --vaapi-device PATH to specify an alternate render node.");
    eprintln!(
        "        Use --backend auto to fall back to CPU silently, \
         or --backend cpu to skip GPU entirely."
    );
}

fn print_vulkan_hint() {
    eprintln!("        --backend vulkan requires a Vulkan 1.3+ device and the loader.");
    eprintln!("        Run `vulkaninfo` to verify a usable Vulkan device is present.");
    eprintln!(
        "        Use --backend auto to fall back to CUDA / CPU silently, \
         or --backend cpu to skip GPU entirely."
    );
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
        eprintln!("rrocket: page {page}: {err}");
        print_error_chain(err);
    }
    std::process::exit(1);
}
