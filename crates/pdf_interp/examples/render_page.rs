//! Render a single PDF page to a PPM file using our native stack (no poppler).
//!
//! Usage: cargo run -p pdf_interp --example render_page -- <pdf> [page] [dpi] [out.ppm]

fn main() {
    let _ = env_logger::try_init();
    let mut args = std::env::args().skip(1);
    let pdf = args
        .next()
        .unwrap_or_else(|| "tests/fixtures/ritual-14th.pdf".to_string());
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
    let page_id = *pages.get(&page).unwrap_or_else(|| {
        eprintln!("page {page} out of range (document has {total} pages)");
        std::process::exit(1);
    });

    let (w_pts, h_pts) = pdf_interp::page_size_pts(&doc, page).expect("page size");
    println!("Page {page} size: {w_pts:.1} × {h_pts:.1} pt");

    let w = (w_pts * dpi / 72.0).round() as u32;
    let h = (h_pts * dpi / 72.0).round() as u32;

    let ops = pdf_interp::parse_page(&doc, page).expect("parse page");
    println!("Page {page}: {} operators", ops.len());

    let scale = dpi / 72.0;
    let mut renderer = pdf_interp::renderer::PageRenderer::new_scaled(w, h, scale, &doc, page_id);
    renderer.execute(&ops);
    let bitmap = renderer.finish();

    let file = std::fs::File::create(&out).expect("create output file");
    let writer = std::io::BufWriter::new(file);
    encode::write_ppm(&bitmap, writer).expect("write PPM");
    println!("Written {out} ({w}×{h} px at {dpi} dpi)");
}
