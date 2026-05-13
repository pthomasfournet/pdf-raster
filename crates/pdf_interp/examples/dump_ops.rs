//! Print every PDF content-stream operator on a page.
//!
//! Usage: `cargo run -p rasterrocket-interp --example dump_ops -- <pdf> [page]`

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .unwrap_or_else(|| "tests/fixtures/input.pdf".to_string());
    let page: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let doc = rasterrocket_interp::open(&path).unwrap_or_else(|e| {
        eprintln!("error: failed to open {path}: {e}");
        std::process::exit(1);
    });
    let ops = rasterrocket_interp::parse_page(&doc, page).unwrap_or_else(|e| {
        eprintln!("error: failed to parse page {page}: {e}");
        std::process::exit(1);
    });
    for op in &ops {
        println!("{op:?}");
    }
}
