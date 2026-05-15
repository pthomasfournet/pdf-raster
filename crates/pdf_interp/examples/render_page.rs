//! Render a single PDF page to a PPM file using the native render stack.
//!
//! Usage: `cargo run -p rasterrocket-interp --example render_page -- <pdf> [page] [dpi] [out.ppm]`

/// Maximum pixel dimension per axis.  Matches `rasterrocket::MAX_PX_DIMENSION`;
/// the example doesn't depend on that crate so the value is duplicated here.
const MAX_PX: f64 = 32_768.0;

fn main() {
    let _ = env_logger::try_init();
    let mut args = std::env::args().skip(1);
    let pdf = args
        .next()
        .unwrap_or_else(|| "tests/fixtures/input.pdf".to_string());
    let page = args.next().and_then(|s| s.parse::<u32>().ok()).unwrap_or(1);
    let dpi = args
        .next()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(72.0);
    let out = args
        .next()
        .unwrap_or_else(|| "/tmp/render_out.ppm".to_string());

    let doc = rasterrocket_interp::open(&pdf).unwrap_or_else(|e| {
        eprintln!("error: failed to open {pdf}: {e}");
        std::process::exit(1);
    });
    let total = rasterrocket_interp::page_count(&doc);
    println!("Opened {pdf}: {total} pages");

    if page == 0 || page > total {
        eprintln!("error: page {page} out of range (document has {total} pages)");
        std::process::exit(1);
    }
    let page_id = doc.get_page(page - 1).unwrap_or_else(|e| {
        eprintln!("error: could not resolve page {page}: {e}");
        std::process::exit(1);
    });

    let geom = rasterrocket_interp::page_size_pts(&doc, page).unwrap_or_else(|e| {
        eprintln!("error: could not determine size of page {page}: {e}");
        std::process::exit(1);
    });
    println!(
        "Page {page} size: {:.1} × {:.1} pt (rotate {}°)",
        geom.width_pts, geom.height_pts, geom.rotate_cw
    );

    let scale = dpi / 72.0;
    // Bound the dimensions before `as u32` so a malformed PageGeometry or an
    // extreme DPI can't silently saturate to u32::MAX.
    let w_f = (geom.width_pts * scale).round();
    let h_f = (geom.height_pts * scale).round();
    if !w_f.is_finite()
        || !h_f.is_finite()
        || w_f <= 0.0
        || h_f <= 0.0
        || w_f > MAX_PX
        || h_f > MAX_PX
    {
        eprintln!(
            "error: page {page} dimensions out of range ({w_f}×{h_f} px at {dpi} dpi; \
             max {MAX_PX} per axis)"
        );
        std::process::exit(1);
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "bounds-checked above: w_f and h_f are finite, > 0, <= MAX_PX (32768)"
    )]
    let (w, h) = (w_f as u32, h_f as u32);

    let ops = rasterrocket_interp::parse_page(&doc, page).unwrap_or_else(|e| {
        eprintln!("error: failed to parse page {page}: {e}");
        std::process::exit(1);
    });
    println!("Page {page}: {} operators", ops.len());

    let mut renderer = rasterrocket_interp::renderer::PageRenderer::new_scaled(
        w,
        h,
        scale,
        geom.rotate_cw,
        geom.origin_x,
        geom.origin_y,
        &doc,
        page_id,
    )
    .unwrap_or_else(|e| {
        eprintln!("error: FreeType initialisation failed: {e}");
        std::process::exit(1);
    });
    renderer.execute(&ops);
    let (bitmap, _diag) = renderer.finish();

    let file = std::fs::File::create(&out).unwrap_or_else(|e| {
        eprintln!("error: could not create output file {out}: {e}");
        std::process::exit(1);
    });
    let writer = std::io::BufWriter::new(file);
    encode::write_ppm(&bitmap, writer).unwrap_or_else(|e| {
        eprintln!("error: failed to write PPM to {out}: {e}");
        std::process::exit(1);
    });
    println!("Written {out} ({w}×{h} px at {dpi} dpi)");
}
