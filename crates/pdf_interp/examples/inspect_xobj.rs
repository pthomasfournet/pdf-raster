fn main() {
    use lopdf::Document;
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tests/fixtures/input.pdf".to_string());
    let doc = Document::load(&path).expect("load PDF");
    let page_ids: Vec<_> = doc.page_iter().collect();
    let page_id = match page_ids.first() {
        Some(id) => *id,
        None => {
            eprintln!("error: document has no pages");
            std::process::exit(1);
        }
    };
    let images = doc.get_page_images(page_id).expect("get page images");
    for img in &images {
        let obj = doc.get_object(img.id).expect("get object");
        let stream = obj.as_stream().expect("object is not a stream");
        println!("Stream dict:");
        for (k, v) in &stream.dict {
            println!("  /{}: {v:?}", String::from_utf8_lossy(k));
        }
    }
}
