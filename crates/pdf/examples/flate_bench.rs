//! Microbench for the FlateDecode backend.
//!
//! Walks a PDF, harvests every `/Filter /FlateDecode` stream, and runs the
//! public `decode_stream` API on each in a hot loop. Compares wall time
//! between feature configurations.
//!
//! Run (libdeflate, default):
//!   cargo run --release -p pdf --example flate_bench -- <path-to-pdf>
//!
//! Run (miniz_oxide fallback):
//!   cargo run --release -p pdf --no-default-features --example flate_bench -- <path-to-pdf>

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use pdf::{Dictionary, Document, Object, Stream};

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/corpus-03-native-text-dense.pdf")
    };

    let bytes = std::fs::read(&path).expect("read pdf");
    let doc = Document::from_bytes_owned(bytes).expect("open pdf");

    // Walk every reachable indirect object and harvest streams whose dict
    // declares FlateDecode (alone or first in a chain).
    let mut samples: Vec<Stream> = Vec::new();
    let mut total_raw_bytes: u64 = 0;
    for (_, page_id) in doc.get_pages() {
        // Page content stream
        if let Ok(page_dict) = doc.get_dict(page_id) {
            collect_from_object(
                &doc,
                page_dict.get(b"Contents"),
                &mut samples,
                &mut total_raw_bytes,
            );
            // Resources/XObject (forms + images; only flate ones survive the filter)
            if let Some(res_obj) = page_dict.get(b"Resources") {
                let res = match res_obj {
                    Object::Dictionary(d) => Some(d.clone()),
                    Object::Reference(rid) => {
                        doc.get_object(*rid).ok().and_then(|o| o.as_dict().cloned())
                    }
                    _ => None,
                };
                if let Some(res) = res
                    && let Some(xobj) = res.get(b"XObject")
                {
                    let xobj_dict = match xobj {
                        Object::Dictionary(d) => Some(d.clone()),
                        Object::Reference(rid) => {
                            doc.get_object(*rid).ok().and_then(|o| o.as_dict().cloned())
                        }
                        _ => None,
                    };
                    if let Some(xobj_dict) = xobj_dict {
                        for (_, v) in xobj_dict.iter() {
                            collect_from_object(&doc, Some(v), &mut samples, &mut total_raw_bytes);
                        }
                    }
                }
            }
        }
    }

    if samples.is_empty() {
        eprintln!("no FlateDecode streams found in {}", path.display());
        std::process::exit(1);
    }

    let backend = if cfg!(feature = "libdeflate") {
        "libdeflate"
    } else {
        "miniz_oxide"
    };
    println!(
        "backend: {backend}; harvested {} flate streams totalling {} raw bytes",
        samples.len(),
        total_raw_bytes
    );

    // Warmup
    let mut warm_total: u64 = 0;
    for s in &samples {
        let out = s.decompressed_content().expect("decode warmup");
        warm_total = warm_total.wrapping_add(out.len() as u64);
    }
    let _ = warm_total;

    let iters: u32 = 20;
    let t0 = Instant::now();
    let mut total_out: u64 = 0;
    for _ in 0..iters {
        for s in &samples {
            let out = s.decompressed_content().expect("decode timed");
            total_out = total_out.wrapping_add(out.len() as u64);
        }
    }
    let elapsed = t0.elapsed();
    let per_iter_ms = (elapsed.as_secs_f64() * 1_000.0) / f64::from(iters);
    let throughput_mibs =
        (total_raw_bytes as f64 * f64::from(iters)) / (1024.0 * 1024.0) / elapsed.as_secs_f64();
    println!(
        "decode_stream: {per_iter_ms:.2} ms/iter ({iters} iters over {} streams; \
         decompressed total {total_out} bytes; raw throughput {throughput_mibs:.1} MiB/s)",
        samples.len()
    );
}

fn collect_from_object(
    doc: &Document,
    obj: Option<&Object>,
    samples: &mut Vec<Stream>,
    total_raw_bytes: &mut u64,
) {
    let Some(obj) = obj else { return };
    match obj {
        Object::Reference(rid) => {
            if let Ok(target) = doc.get_object(*rid) {
                collect_from_object(doc, Some(target.as_ref()), samples, total_raw_bytes);
            }
        }
        Object::Array(arr) => {
            for it in arr {
                collect_from_object(doc, Some(it), samples, total_raw_bytes);
            }
        }
        Object::Stream(stream) if first_filter_is_flate(&stream.dict) => {
            *total_raw_bytes = total_raw_bytes.wrapping_add(stream.content.len() as u64);
            samples.push(Stream::new(stream.dict.clone(), stream.content.clone()));
        }
        _ => {}
    }
}

fn first_filter_is_flate(dict: &Dictionary) -> bool {
    match dict.get(b"Filter") {
        Some(Object::Name(n)) => n == b"FlateDecode",
        Some(Object::Array(a)) => match a.first() {
            Some(Object::Name(n)) => n == b"FlateDecode",
            _ => false,
        },
        _ => false,
    }
}
