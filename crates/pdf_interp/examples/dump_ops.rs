//! Print every PDF content-stream operator on a page.
//!
//! Usage: `cargo run -p pdf_interp --example dump_ops -- <pdf> [page]`

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .unwrap_or_else(|| "tests/fixtures/input.pdf".to_string());
    let page: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let doc = pdf_interp::open(&path).unwrap();
    let ops = pdf_interp::parse_page(&doc, page).unwrap();
    for op in &ops {
        println!("{op:?}");
    }
}
