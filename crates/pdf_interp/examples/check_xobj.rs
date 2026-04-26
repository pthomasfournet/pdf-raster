fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tests/fixtures/scotch-rite-illustrated.pdf".to_string());
    let page: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let doc = pdf_interp::open(&path).unwrap();
    let pages = doc.get_pages();
    let page_id = *pages.get(&page).unwrap();
    let page_dict = doc.get_dictionary(page_id).unwrap();

    use lopdf::{Document, Object};
    fn resolve_dict<'a>(doc: &'a Document, obj: &'a Object) -> Option<&'a lopdf::Dictionary> {
        match obj {
            Object::Dictionary(d) => Some(d),
            Object::Reference(id) => doc.get_dictionary(*id).ok(),
            _ => None,
        }
    }

    let res_obj = page_dict.get(b"Resources").unwrap();
    let res = resolve_dict(&doc, res_obj).unwrap();
    if let Ok(xobj_obj) = res.get(b"XObject") {
        let xobj = resolve_dict(&doc, xobj_obj).unwrap();
        for (k, v) in xobj {
            let name = String::from_utf8_lossy(k);
            if let Object::Reference(id) = v {
                let stream = doc
                    .get_object(*id)
                    .and_then(|o| o.as_stream().map(|s| s.clone()))
                    .ok();
                if let Some(s) = stream {
                    let filter = s
                        .dict
                        .get(b"Filter")
                        .ok()
                        .map(|f| format!("{f:?}"))
                        .unwrap_or("None".into());
                    let subtype = s
                        .dict
                        .get(b"Subtype")
                        .ok()
                        .map(|f| format!("{f:?}"))
                        .unwrap_or("None".into());
                    let w = s
                        .dict
                        .get(b"Width")
                        .ok()
                        .map(|v| format!("{v:?}"))
                        .unwrap_or("?".into());
                    let h = s
                        .dict
                        .get(b"Height")
                        .ok()
                        .map(|v| format!("{v:?}"))
                        .unwrap_or("?".into());
                    println!("/{name}: Subtype={subtype} Filter={filter} W={w} H={h}");
                }
            }
        }
    }
}
