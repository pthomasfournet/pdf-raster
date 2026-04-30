//! Render a single PDF page to a PPM file using our native stack (no poppler).
//!
//! Usage: cargo run -p pdf_interp --example render_page -- <pdf> [page] [dpi] [out.ppm]

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

    let doc = pdf_interp::open(&pdf).expect("open PDF");
    let total = pdf_interp::page_count(&doc);
    println!("Opened {pdf}: {total} pages");

    let pages = doc.get_pages();
    let page_id = match pages.get(&page) {
        Some(id) => *id,
        None => {
            eprintln!("error: page {page} out of range (document has {total} pages)");
            std::process::exit(1);
        }
    };

    let geom = pdf_interp::page_size_pts(&doc, page).unwrap_or_else(|e| {
        eprintln!("error: could not determine size of page {page}: {e}");
        std::process::exit(1);
    });
    println!(
        "Page {page} size: {:.1} × {:.1} pt (rotate {}°)",
        geom.width_pts, geom.height_pts, geom.rotate_cw
    );

    let scale = dpi / 72.0;
    let w = (geom.width_pts * scale).round() as u32;
    let h = (geom.height_pts * scale).round() as u32;
    if w == 0 || h == 0 {
        eprintln!("error: page {page} has degenerate dimensions ({w}×{h} px at {dpi} dpi)");
        std::process::exit(1);
    }

    let ops = pdf_interp::parse_page(&doc, page).unwrap_or_else(|e| {
        eprintln!("error: failed to parse page {page}: {e}");
        std::process::exit(1);
    });
    println!("Page {page}: {} operators", ops.len());

    let mut renderer =
        pdf_interp::renderer::PageRenderer::new_scaled(w, h, scale, geom.rotate_cw, &doc, page_id);
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
