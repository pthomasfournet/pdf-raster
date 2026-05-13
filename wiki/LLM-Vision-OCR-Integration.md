# LLM Vision OCR Integration

pdf-raster outputs 8-bit greyscale pixel buffers (`RenderedPage`) that can be encoded and sent to cloud vision APIs. This page covers **Google Cloud Vision** and **OpenAI GPT-5** — the two most relevant cloud APIs for PDF OCR pipelines.

For local, offline OCR without network calls or per-page costs, see [OCR Integration](OCR-Integration).

---

## When to use cloud vs local OCR

| | Local (Tesseract / ocrs) | Google Cloud Vision | GPT-5 | Mistral |
|---|---|---|---|---|
| **Latency** | < 1 s/page | 200–500 ms/page (network) | 1–5 s/page (network) | 2–4 s/page (vision); ~0.5 s/page (OCR 3) |
| **Cost** | Free (compute only) | ~$1.50 / 1000 pages | ~$2–4 / 1000 pages (token-based, varies by page complexity) | $2.00 / 1000 pages (OCR 3); $0.50/M input tokens (Large 3) |
| **Privacy** | Document stays on-machine | Bytes sent to Google | Bytes sent to OpenAI | Bytes sent to Mistral; self-host option via vLLM |
| **Bounding boxes** | Yes (Tesseract) / No (ocrs) | Yes — word/line/block/paragraph | No — text dump only | OCR 3: yes (bbox per word); Large 3: no |
| **Layout preservation** | Partial | Excellent | Good (reasoning-based) | OCR 3: Excellent; Large 3: Good |
| **Handwriting** | Poor–Fair | Good | Excellent | OCR 3: Excellent (88.9%) |
| **Complex / ambiguous content** | Literal OCR only | Literal OCR only | Can reason about context | Large 3: Can reason about context |
| **Rate limits** | None | 1800 req/min (default) | Much lower | Tier-based; check Mistral AI Studio |
| **Best for** | High-throughput batch, air-gapped | Structured docs needing layout | Ambiguous content, handwriting | Budget batch OCR (OCR 3); or self-hosted via vLLM |

> **Privacy note:** When using cloud APIs, document bytes leave your machine and are processed on third-party servers. Review your data processing agreements before sending sensitive content.

---

## Shared encoding helper: `RenderedPage` → base64 JPEG

Both cloud APIs accept base64-encoded image bytes. JPEG is preferred over PNG — smaller payload means lower latency. pdf-raster's `encode` crate does not produce JPEG; add the `image` crate for this step.

```toml
# Cargo.toml
[dependencies]
image = { version = "0.25", default-features = false, features = ["jpeg"] }
base64 = "0.22"
```

**Rust:**

```rust
use image::{GrayImage, ImageEncoder};
use image::codecs::jpeg::JpegEncoder;
use pdf_raster::RenderedPage;

/// Encode a greyscale RenderedPage to a base64 JPEG string.
/// quality: 0–100; 85 is a good default for OCR (small payload, no perceptible loss).
fn page_to_base64_jpeg(page: &RenderedPage, quality: u8) -> anyhow::Result<String> {
    let img = GrayImage::from_raw(page.width, page.height, page.pixels.clone())
        .ok_or_else(|| anyhow::anyhow!("invalid pixel buffer dimensions"))?;

    let mut jpeg_buf = Vec::new();
    let encoder = JpegEncoder::new_with_quality(&mut jpeg_buf, quality);
    encoder.encode(
        img.as_raw(),
        page.width,
        page.height,
        image::ColorType::L8,
    )?;

    Ok(base64::engine::general_purpose::STANDARD.encode(&jpeg_buf))
}
```

**Python:**

```python
import base64
import io
from PIL import Image

def page_to_base64_jpeg(pixels: bytes, width: int, height: int, quality: int = 85) -> str:
    """Encode a greyscale pixel buffer to base64 JPEG."""
    img = Image.frombytes("L", (width, height), pixels)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()
```

---

## Google Cloud Vision

### Pricing and limits (as of May 2026)

- **DOCUMENT_TEXT_DETECTION:** ~$1.50 / 1000 pages
- **Default rate limit:** 1800 requests/min — parallelise freely up to this limit
- Request a quota increase via the Google Cloud Console for large batches
- Prices change — verify at https://cloud.google.com/vision/pricing

### What you get

`DOCUMENT_TEXT_DETECTION` returns full structured output per page:
- Paragraph, block, line, word, and symbol boundaries
- Bounding boxes for every element
- Per-symbol confidence scores
- Language detection per block

Use this when layout, reading order, or bounding-box coordinates matter downstream.

### Setup

```bash
# Install Google Cloud SDK
gcloud auth application-default login

# Or set credentials directly
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

```toml
# Cargo.toml
[dependencies]
reqwest = { version = "0.12", features = ["json", "rustls-tls"] }
tokio = { version = "1", features = ["full"] }
serde_json = "1"
futures = "0.3"
image = { version = "0.25", default-features = false, features = ["jpeg"] }
base64 = "0.22"
```

### Rust example — parallel batch

```rust
use pdf_raster::{RasterOptions, render_channel};
use std::path::Path;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = std::env::var("GOOGLE_CLOUD_VISION_API_KEY")?;
    let client = reqwest::Client::new();

    let opts = RasterOptions { dpi: 300.0, first_page: 1, last_page: u32::MAX, deskew: false, pages: None };
    // Collect pages so we can spawn async tasks.
    let pages: Vec<_> = render_channel(Path::new("scan.pdf"), &opts, 8)
        .iter()
        .collect();

    let mut tasks = Vec::new();
    for (page_num, result) in pages {
        let page = result?;
        let b64 = page_to_base64_jpeg(&page, 85)?;
        let client = client.clone();
        let api_key = api_key.clone();

        tasks.push(tokio::spawn(async move {
            let body = serde_json::json!({
                "requests": [{
                    "image": { "content": b64 },
                    "features": [{ "type": "DOCUMENT_TEXT_DETECTION" }]
                }]
            });
            let resp = client
                .post(format!(
                    "https://vision.googleapis.com/v1/images:annotate?key={api_key}"
                ))
                .json(&body)
                .send()
                .await?
                .json::<serde_json::Value>()
                .await?;

            let text = resp["responses"][0]["fullTextAnnotation"]["text"]
                .as_str()
                .unwrap_or("")
                .to_string();

            anyhow::Ok((page_num, text))
        }));
    }

    let mut results: Vec<(u32, String)> = futures::future::join_all(tasks)
        .await
        .into_iter()
        .filter_map(|r| r.ok()?.ok())
        .collect();
    results.sort_by_key(|(n, _)| *n);

    for (page_num, text) in &results {
        println!("=== Page {page_num} ===\n{text}\n");
    }
    Ok(())
}
```

> **Tip:** For very large documents, batch requests into groups of 16 (Cloud Vision API limit per `annotate` call) and use a semaphore to cap concurrent requests below the rate limit.

### Python example

```python
import io, concurrent.futures
from google.cloud import vision
from PIL import Image

def ocr_page(client, pixels: bytes, width: int, height: int) -> str:
    img = Image.frombytes("L", (width, height), pixels)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)

    response = client.document_text_detection(
        image=vision.Image(content=buf.getvalue())
    )
    return response.full_text_annotation.text

def ocr_pdf_cloud(pages):
    """pages: list of (page_num, pixels, width, height, effective_dpi)"""
    client = vision.ImageAnnotatorClient()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures = {
            pool.submit(ocr_page, client, px, w, h): n
            for n, px, w, h, _ in pages
        }
        results = {}
        for fut in concurrent.futures.as_completed(futures):
            page_num = futures[fut]
            results[page_num] = fut.result()
    return [results[n] for n in sorted(results)]
```

---

## GPT-5 (OpenAI vision)

### When to use GPT-5 over Cloud Vision

- Document content is ambiguous or requires contextual reasoning
- Handwriting is the primary content
- You need answers to questions about the document, not just transcription
- Accuracy on a few pages matters more than cost or throughput

**Do not use GPT-5 for high-throughput batch OCR** — latency (1–5 s/page) and higher cost make it unsuitable for large batches.

### What you get

GPT-5 returns a text completion — no bounding boxes, no confidence scores, no structured layout. If you need coordinates or word-level structure, use Cloud Vision instead.

### Setup

```bash
export OPENAI_API_KEY=sk-...
```

```toml
# Cargo.toml — same deps as Cloud Vision section above
[dependencies]
reqwest = { version = "0.12", features = ["json", "rustls-tls"] }
tokio = { version = "1", features = ["full"] }
serde_json = "1"
image = { version = "0.25", default-features = false, features = ["jpeg"] }
base64 = "0.22"
```

### Rust example

```rust
use pdf_raster::{RasterOptions, raster_pdf};
use std::path::Path;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = std::env::var("OPENAI_API_KEY")?;
    let client = reqwest::Client::new();

    let opts = RasterOptions { dpi: 300.0, first_page: 1, last_page: u32::MAX, deskew: false, pages: None };

    for (page_num, result) in raster_pdf(Path::new("scan.pdf"), &opts) {
        let page = result?;
        let b64 = page_to_base64_jpeg(&page, 85)?;

        let body = serde_json::json!({
            "model": "gpt-5",
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": format!("data:image/jpeg;base64,{b64}"),
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Transcribe all text in this document image exactly as it appears."
                    }
                ]
            }],
            "max_tokens": 4096
        });

        let resp = client
            .post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(&api_key)
            .json(&body)
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        let text = resp["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        println!("=== Page {page_num} ===\n{text}\n");
    }
    Ok(())
}
```

### Python example

```python
import base64, io
from openai import OpenAI
from PIL import Image

client = OpenAI()  # reads OPENAI_API_KEY from env

def ocr_page_gpt(pixels: bytes, width: int, height: int) -> str:
    img = Image.frombytes("L", (width, height), pixels)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()

    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "high"
                    }
                },
                {
                    "type": "text",
                    "text": "Transcribe all text in this document image exactly as it appears."
                }
            ]
        }],
        max_tokens=4096,
    )
    return response.choices[0].message.content
```

> **Rate limits:** GPT-5 has significantly lower rate limits than Cloud Vision. For batches larger than a few dozen pages, add exponential backoff and respect `Retry-After` headers. Consider Cloud Vision instead for large-volume work.

---

## Mistral (Large 3 / OCR 3)

Mistral offers two distinct paths for PDF OCR pipelines:

- **Mistral OCR 3** — a dedicated document OCR API, optimised for speed and accuracy on scanned documents, tables, and handwriting. Returns bounding boxes. $2.00/1000 pages standard; $1.00/1000 pages via Batch API.
- **Mistral Large 3** (`mistral-large-2512`) — general-purpose vision model for mixed reasoning + OCR tasks. Same use case as GPT-5 but cheaper per token.

Both use the same OpenAI-compatible endpoint, so the code is nearly identical to the GPT-5 examples above.

### Mistral OCR 3 — when to use

- High-volume batch OCR where cost and speed matter
- Scanned documents with degraded quality
- Tables, forms, and handwriting (88.9% handwriting accuracy)
- You need bounding boxes without Cloud Vision's price

### Mistral Large 3 — when to use

- Mixed tasks: transcribe text AND reason about content
- Drop-in alternative to GPT-5 at lower cost
- Self-hosted option needed (use Pixtral 12B via vLLM instead — see below)

### Setup

```bash
export MISTRAL_API_KEY=...
```

```toml
# Cargo.toml — same base deps as Cloud Vision / GPT-5 sections
[dependencies]
reqwest = { version = "0.12", features = ["json", "rustls-tls"] }
tokio = { version = "1", features = ["full"] }
serde_json = "1"
image = { version = "0.25", default-features = false, features = ["jpeg"] }
base64 = "0.22"
```

### Rust example — Mistral OCR 3

```rust
use pdf_raster::{RasterOptions, raster_pdf};
use std::path::Path;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = std::env::var("MISTRAL_API_KEY")?;
    let client = reqwest::Client::new();

    let opts = RasterOptions { dpi: 300.0, first_page: 1, last_page: u32::MAX, deskew: false, pages: None };

    for (page_num, result) in raster_pdf(Path::new("scan.pdf"), &opts) {
        let page = result?;
        let b64 = page_to_base64_jpeg(&page, 85)?;

        let body = serde_json::json!({
            "model": "mistral-ocr-latest",
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": format!("data:image/jpeg;base64,{b64}")
                        }
                    }
                ]
            }]
        });

        let resp = client
            .post("https://api.mistral.ai/v1/chat/completions")
            .bearer_auth(&api_key)
            .json(&body)
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        let text = resp["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        println!("=== Page {page_num} ===\n{text}\n");
    }
    Ok(())
}
```

### Rust example — Mistral Large 3 (vision + reasoning)

```rust
// Same structure as OCR 3 above — only the model name and prompt change.
let body = serde_json::json!({
    "model": "mistral-large-2512",
    "messages": [{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": format!("data:image/jpeg;base64,{b64}")
                }
            },
            {
                "type": "text",
                "text": "Transcribe all text in this document image exactly as it appears."
            }
        ]
    }],
    "max_tokens": 4096
});
```

### Python example

```python
import base64, io
from mistralai import Mistral
from PIL import Image

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

def ocr_page_mistral(pixels: bytes, width: int, height: int, model: str = "mistral-ocr-latest") -> str:
    img = Image.frombytes("L", (width, height), pixels)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()

    response = client.chat.complete(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": "Transcribe all text in this document image exactly as it appears."}
            ]
        }]
    )
    return response.choices[0].message.content
```

### Self-hosted via vLLM (Pixtral 12B)

Pixtral 12B is open-weight (Apache 2.0) and runs locally via vLLM — no API key, no per-page cost.

```bash
pip install vllm
# Requires HuggingFace token for model download
huggingface-cli login
vllm serve mistralai/Pixtral-12B-2409 --tokenizer_mode mistral --config_format mistral --load_format mistral
```

Once running, point the Rust or Python examples at `http://localhost:8000/v1/chat/completions` with `bearer_auth("token")` and model `"mistralai/Pixtral-12B-2409"`. No other code changes needed.

> **Note:** Pixtral 12B requires ~24 GB VRAM. Pixtral Large (124B) requires multi-GPU. For CPU-only inference, use Tesseract or ocrs instead — see [OCR Integration](OCR-Integration).
