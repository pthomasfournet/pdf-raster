# OCR Integration

pdf-raster outputs 8-bit greyscale pixel buffers (`RenderedPage`) ready for direct consumption by any OCR engine. This page documents the optimal integration pattern for two Rust-native, offline engines: **Tesseract** (via `leptess`) and **ocrs**.

For cloud vision APIs (Google Cloud Vision, GPT-5), see [LLM Vision OCR Integration](LLM-Vision-OCR-Integration).

---

## Quick comparison

| | Tesseract (`leptess`) | ocrs |
|---|---|---|
| **Accuracy** | Excellent — best on clean documents, wide language support | Good — strong on printed English, weaker on complex scripts |
| **System deps** | Requires `libtesseract` + language data files installed | Pure Rust — no system deps, WASM-compatible |
| **Zero-copy ingestion** | Yes — `set_image_from_mem` aliases your buffer | Near-zero — one `u8→f32` normalisation pass, no extra copies |
| **Threading model** | One `TessApi` per thread (not `Send`) | Single `Arc<OcrEngine>` shared across threads (`Send + Sync`) |
| **DPI required** | Yes — pass `page.effective_dpi` or accuracy degrades | No — character size inferred from pixel dimensions |

---

## DPI guidance

`RenderedPage` carries two DPI fields:

- `dpi` — the render resolution you passed in `RasterOptions`
- `effective_dpi` — `dpi × UserUnit`, accounting for PDF `UserUnit` scaling

**Always pass `effective_dpi` to downstream consumers.** For the vast majority of PDFs `UserUnit` is 1.0 and both fields are equal, but when they differ, using `dpi` instead of `effective_dpi` misreports the resolution and degrades OCR accuracy.

Recommended render DPI for OCR: **300**. Character x-height should be 20–30 px at that resolution. 150 DPI is acceptable for large-print documents; below 150 DPI accuracy drops sharply.

---

## Tesseract via `leptess`

### Add the dependency

```toml
# Cargo.toml
[dependencies]
leptess = "0.14"
```

Requires `libtesseract-dev` and at least one language data package installed on the system:

```bash
# Debian / Ubuntu
sudo apt install libtesseract-dev tesseract-ocr-eng

# macOS
brew install tesseract
```

### The footgun to avoid

The `tesseract` 0.15.x crate's convenience function creates a **new `TessBaseAPI` instance per call**:

```rust
// ❌ DO NOT USE IN A PIPELINE — ~100 ms init overhead per page
let text = tesseract::ocr_from_frame(&page.pixels, ...)?;
```

Instance initialisation loads model weights from disk on every call. At 100 ms per page, a 1000-page document wastes ~100 seconds just on reinitialisation.

### Correct pattern — single-threaded

Create one `TessApi` and reuse it across all pages:

```rust
use leptess::{LepTess, Variable};
use pdf_raster::{RasterOptions, raster_pdf};
use std::path::Path;

fn ocr_pdf(path: &Path, dpi: f32) -> anyhow::Result<Vec<String>> {
    // Create once. Model weights loaded here, not per page.
    let mut api = LepTess::new(None, "eng")?;

    let opts = RasterOptions { dpi, first_page: 1, last_page: u32::MAX, deskew: false, pages: None };
    let mut results = Vec::new();

    for (_page_num, result) in raster_pdf(path, &opts) {
        let page = result?;

        // Set DPI once (or per page if your PDF has variable UserUnit).
        // effective_dpi accounts for UserUnit scaling — always use it.
        api.set_source_resolution(page.effective_dpi as i32);

        // Zero-copy: set_image_from_mem aliases page.pixels — no copy.
        // page.pixels must stay alive until get_utf8_text() returns.
        api.set_image_from_mem(&page.pixels)?;
        api.set_variable(Variable::TesseditCreateBoxfile, "0")?;

        results.push(api.get_utf8_text()?);
    }

    Ok(results)
}
```

### Correct pattern — multi-threaded

`leptess::LepTess` is not `Send`. Spawn one thread per worker, each with its own instance. Feed pages via `render_channel`:

```rust
use leptess::LepTess;
use pdf_raster::{RasterOptions, render_channel};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;

fn ocr_pdf_parallel(path: &Path, dpi: f32, workers: usize) -> anyhow::Result<Vec<(u32, String)>> {
    let opts = RasterOptions { dpi, first_page: 1, last_page: u32::MAX, deskew: false, pages: None };
    let rx = Arc::new(Mutex::new(render_channel(path, &opts, workers * 2)));

    let (tx_out, rx_out) = std::sync::mpsc::channel();

    // Distribute work across N threads, each holding its own TessApi.
    let handles: Vec<_> = (0..workers)
        .map(|_| {
            let rx = Arc::clone(&rx);
            let tx_out = tx_out.clone();
            thread::spawn(move || {
                // Each thread creates its own instance — never shared.
                let mut api = LepTess::new(None, "eng").expect("leptess init");
                loop {
                    let item = rx.lock().unwrap().recv();
                    let Ok((page_num, result)) = item else { break };
                    let page = match result {
                        Ok(p) => p,
                        Err(e) => { eprintln!("page {page_num}: {e}"); continue; }
                    };
                    api.set_source_resolution(page.effective_dpi as i32);
                    api.set_image_from_mem(&page.pixels).expect("set_image");
                    let text = api.get_utf8_text().unwrap_or_default();
                    tx_out.send((page_num, text)).ok();
                }
            })
        })
        .collect();

    drop(tx_out);
    let mut results: Vec<(u32, String)> = rx_out.iter().collect();
    for h in handles { h.join().ok(); }
    results.sort_by_key(|(n, _)| *n);
    Ok(results)
}
```

> **Note:** `render_channel` returns `std::sync::mpsc::Receiver`, which is not `Clone`. The example above wraps it in `Arc<Mutex<_>>` so multiple threads can drain the same channel. For higher throughput, `crossbeam-channel` eliminates the per-recv mutex.

---

## ocrs

### Add the dependency

```toml
# Cargo.toml
[dependencies]
ocrs = "0.7"
rten = "0.13"        # ocrs runtime dependency
```

Download model weights (run once):

```bash
# Detection model
curl -L https://ocrs-models.s3-accelerate.amazonaws.com/text-detection.rten -o text-detection.rten
# Recognition model
curl -L https://ocrs-models.s3-accelerate.amazonaws.com/text-recognition.rten -o text-recognition.rten
```

### The footgun to avoid

Loading model weights inside the page loop is expensive (~seconds on first call):

```rust
// ❌ DO NOT DO THIS — reloads models on every page
for page in pages {
    let engine = OcrEngine::new(OcrEngineParams { .. })?;  // expensive
    engine.get_text(&input)?;
}
```

### Correct pattern — shared engine

`OcrEngine` is `Send + Sync`. Construct once, wrap in `Arc`, share across threads:

```rust
use ocrs::{OcrEngine, OcrEngineParams, ImageSource};
use pdf_raster::{RasterOptions, raster_pdf};
use rten::Model;
use std::path::Path;
use std::sync::Arc;

fn ocr_pdf(path: &Path, dpi: f32) -> anyhow::Result<Vec<String>> {
    // Load models once. This is the expensive step (~100s ms).
    let detection_model = Model::load_file("text-detection.rten")?;
    let recognition_model = Model::load_file("text-recognition.rten")?;

    let engine = Arc::new(OcrEngine::new(OcrEngineParams {
        detection_model: Some(detection_model),
        recognition_model: Some(recognition_model),
        ..Default::default()
    })?);

    let opts = RasterOptions { dpi, first_page: 1, last_page: u32::MAX, deskew: false, pages: None };
    let mut results = Vec::new();

    for (_page_num, result) in raster_pdf(path, &opts) {
        let page = result?;

        // Zero allocation: from_bytes creates a tensor view over page.pixels.
        // One f32 buffer is allocated inside prepare_input (u8→f32 normalisation).
        let source = ImageSource::from_bytes(&page.pixels, (page.width, page.height))?;
        let input = engine.prepare_input(source)?;

        let words = engine.detect_words(&input)?;
        let lines = engine.find_text_lines(&input, &words);
        let text_items = engine.recognize_text(&input, &lines)?;

        let text = text_items
            .iter()
            .filter_map(|item| item.as_ref())
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        results.push(text);
    }

    Ok(results)
}
```

### Multi-threaded pattern

```rust
use ocrs::{OcrEngine, OcrEngineParams, ImageSource};
use pdf_raster::{RasterOptions, render_channel};
use rten::Model;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;

fn ocr_pdf_parallel(path: &Path, dpi: f32, workers: usize) -> anyhow::Result<Vec<(u32, String)>> {
    let detection_model = Model::load_file("text-detection.rten")?;
    let recognition_model = Model::load_file("text-recognition.rten")?;
    let engine = Arc::new(OcrEngine::new(OcrEngineParams {
        detection_model: Some(detection_model),
        recognition_model: Some(recognition_model),
        ..Default::default()
    })?);

    let opts = RasterOptions { dpi, first_page: 1, last_page: u32::MAX, deskew: false, pages: None };
    let rx = Arc::new(Mutex::new(render_channel(path, &opts, workers * 2)));
    let (tx_out, rx_out) = std::sync::mpsc::channel();

    let handles: Vec<_> = (0..workers)
        .map(|_| {
            let engine = Arc::clone(&engine);
            let rx = Arc::clone(&rx);
            let tx_out = tx_out.clone();
            thread::spawn(move || {
                loop {
                    let item = rx.lock().unwrap().recv();
                    let Ok((page_num, result)) = item else { break };
                    let page = match result {
                        Ok(p) => p,
                        Err(e) => { eprintln!("page {page_num}: {e}"); continue; }
                    };
                    let source = ImageSource::from_bytes(
                        &page.pixels, (page.width, page.height)
                    ).expect("ImageSource");
                    let input = engine.prepare_input(source).expect("prepare_input");
                    let words = engine.detect_words(&input).expect("detect_words");
                    let lines = engine.find_text_lines(&input, &words);
                    let text_items = engine.recognize_text(&input, &lines).expect("recognize");
                    let text = text_items.iter()
                        .filter_map(|i| i.as_ref())
                        .map(|line| line.to_string())
                        .collect::<Vec<_>>()
                        .join("\n");
                    tx_out.send((page_num, text)).ok();
                }
            })
        })
        .collect();

    drop(tx_out);
    let mut results: Vec<(u32, String)> = rx_out.iter().collect();
    for h in handles { h.join().ok(); }
    results.sort_by_key(|(n, _)| *n);
    Ok(results)
}
```

> **Note:** ocrs uses Rayon internally for detection and layout parallelism. Setting `workers` above the number of physical cores may cause over-subscription. Start with `workers = num_cpus::get() / 2` and tune from there.

---

## Choosing between Tesseract and ocrs

| Situation | Recommendation |
|-----------|---------------|
| Need wide language support or high accuracy on degraded scans | Tesseract |
| Deploying to a container or WASM with no system deps | ocrs |
| Need word-level bounding boxes | Tesseract (`get_component_images`) |
| Simplest possible Rust-only build | ocrs |
| Mixed scripts (Arabic, CJK, RTL) | Tesseract |
