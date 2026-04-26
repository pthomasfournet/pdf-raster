fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tests/fixtures/ritual-14th.pdf".to_string());
    let page: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let doc = pdf_interp::open(&path).unwrap();
    let ops = pdf_interp::parse_page(&doc, page).unwrap();
    for op in &ops {
        println!("{op:?}");
    }
}
