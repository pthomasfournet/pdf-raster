//! Sanity check: parse page 1 of a PDF and report any unknown operators.
//!
//! Usage: `cargo run -p rasterrocket-interp --example smoke -- <pdf>`

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tests/fixtures/input.pdf".to_string());
    let doc = rasterrocket_interp::open(&path).unwrap_or_else(|e| {
        eprintln!("error: failed to open {path}: {e}");
        std::process::exit(1);
    });
    let total = rasterrocket_interp::page_count(&doc);
    println!("Pages: {total}");
    let ops = rasterrocket_interp::parse_page(&doc, 1).unwrap_or_else(|e| {
        eprintln!("error: failed to parse page 1: {e}");
        std::process::exit(1);
    });
    println!("Page 1: {} operators", ops.len());
    for op in ops.iter().take(20) {
        println!("  {op:?}");
    }
    let mut unknown: Vec<_> = ops
        .iter()
        .filter_map(|o| {
            if let rasterrocket_interp::content::Operator::Unknown(kw) = o {
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
