//! Scan every page of a PDF for non-trivial fill colours and report.
//!
//! Usage: `cargo run -p pdf_interp --example scan_fills -- <pdf>`

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tests/fixtures/input.pdf".to_string());
    let doc = pdf_interp::open(&path).unwrap();
    let n = pdf_interp::page_count(&doc);
    for p in 1..=n {
        let ops = pdf_interp::parse_page(&doc, p).unwrap();
        let dark_fills: Vec<_> = ops
            .iter()
            .filter(|o| {
                matches!(o,
                    pdf_interp::content::Operator::SetFillGray(g) if *g < 0.9
                )
            })
            .collect();
        let has_rect = ops
            .iter()
            .any(|o| matches!(o, pdf_interp::content::Operator::Rectangle(..)));
        let has_rgb = ops.iter().any(|o| {
            matches!(
                o,
                pdf_interp::content::Operator::SetFillRgb(..)
                    | pdf_interp::content::Operator::SetStrokeRgb(..)
            )
        });
        if !dark_fills.is_empty() || has_rgb {
            println!(
                "page {p}: dark_fills={} has_rect={has_rect} has_rgb={has_rgb} total_ops={}",
                dark_fills.len(),
                ops.len()
            );
            for f in &dark_fills {
                println!("  {f:?}");
            }
        }
    }
}
