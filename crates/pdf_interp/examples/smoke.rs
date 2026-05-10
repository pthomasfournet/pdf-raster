//! Sanity check: parse page 1 of a PDF and report any unknown operators.
//!
//! Usage: `cargo run -p pdf_interp --example smoke -- <pdf>`

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tests/fixtures/input.pdf".to_string());
    let doc = pdf_interp::open(&path).expect("open");
    let total = pdf_interp::page_count(&doc);
    println!("Pages: {total}");
    let ops = pdf_interp::parse_page(&doc, 1).expect("parse page 1");
    println!("Page 1: {} operators", ops.len());
    for op in ops.iter().take(20) {
        println!("  {op:?}");
    }
    let mut unknown: Vec<_> = ops
        .iter()
        .filter_map(|o| {
            if let pdf_interp::content::Operator::Unknown(kw) = o {
                Some(String::from_utf8_lossy(kw).to_string())
            } else {
                None
            }
        })
        .collect();
    unknown.sort();
    unknown.dedup();
    if unknown.is_empty() {
        println!("No unknown operators.");
    } else {
        println!("Unknown operators: {unknown:?}");
    }
}
