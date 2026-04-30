fn main() {
    use lopdf::Document;
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tests/fixtures/cryptic-rite.pdf".to_string());
    let doc = Document::load(&path).expect("load");
    let page_ids: Vec<_> = doc.page_iter().collect();
    let page_id = page_ids[0];
    let images = doc.get_page_images(page_id).expect("images");
    for img in &images {
        let obj = doc.get_object(img.id).expect("obj");
        let stream = obj.as_stream().expect("stream");
        println!("Stream dict:");
        for (k, v) in &stream.dict {
            println!("  /{}: {v:?}", String::from_utf8_lossy(k));
        }
    }
}
