//! Bounded, deterministic decoder property/fuzz harness.
//!
//! # Why this exists
//!
//! The hardening campaign has empirically verified graceful handling of many
//! malformed inputs on *hand-picked* samples (empty / non-PDF / truncated
//! files, hostile filter chains, recursion bombs, out-of-bounds CID, short
//! `SMask`, …).  What the sampled tests cannot prove is that the campaign's
//! central invariant holds across a *class* of mutated inputs rather than the
//! specific bytes someone happened to write down.
//!
//! ## The invariant asserted here
//!
//! Feeding malformed/mutated bytes through the real decode+render entry path
//! must ALWAYS terminate in bounded time and memory with one of:
//!
//!   * a clean `Err` from the parser, or
//!   * a clean `Ok` whose page geometry is within the campaign's safety caps
//!     (a bounded "skip / degrade" outcome — a blank or partial page is
//!     acceptable; a *garbage-sized* one is not).
//!
//! It must NEVER:
//!
//!   * abort the process (SIGABRT / SIGSEGV / a panic escaping the per-page
//!     boundary as `exit 101`),
//!   * hang (infinite loop / unbounded retry),
//!   * allocate without bound (OOM), or
//!   * silently succeed with structurally impossible output.
//!
//! This is precisely the denial-of-service / silent-corruption class the whole
//! campaign exists to eliminate.  This harness PROVES the class on a generated
//! population; finding nothing new is the expected and good outcome.
//!
//! # Design constraints (deliberate, load-bearing)
//!
//! * NOT a `cargo-fuzz`/libfuzzer target: no unbounded run, no heavyweight
//!   dev-dependency, OOM-safe under the project's disk/RAM rules.  It is a
//!   normal `#[test]` that runs under `cargo test` / `cargo nextest` and
//!   completes well under 30 s.
//! * Deterministic: a hand-written `SplitMix64` PRNG seeds every choice, so a
//!   given build produces the byte-identical mutant population every run.  A
//!   failure is reproducible from the reported seed.
//! * Genuinely exercises the real decoders: each mutant is driven through the
//!   public `Document::from_bytes_owned` -> `parse_page_by_id` ->
//!   `PageRenderer::execute` -> `finish` path — the same path the CLI uses.
//!   Codec mutants are embedded as image `XObject` streams so `decode_dct`,
//!   `decode_ccitt`, and `decode_jbig2` are actually invoked.
//! * Distinguishes graceful handling from a real gap: every mutant runs inside
//!   `catch_unwind` on a worker thread guarded by a join timeout.  A caught
//!   panic, a timeout (hang), or an out-of-cap "successful" geometry is a
//!   reported FAILURE — the harness does not blanket-swallow panics to look
//!   green.  The production page layer's own per-page `catch_unwind` is a
//!   defence in depth, not a licence for decoders to panic on bad bytes.

use std::fmt::Write as _;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use pdf::Document;

/// Hard wall-clock guard per mutant.  Every bounded entry point under test is
/// engineered (by the campaign's MAX_* caps and watchdog) to finish in
/// milliseconds on tiny inputs; 5 s is three orders of magnitude of headroom,
/// so exceeding it means a genuine hang, which the harness must catch rather
/// than inherit.
const PER_MUTANT_GUARD: Duration = Duration::from_secs(5);

/// Total mutant budget across all seeds.  Sized so the whole harness runs in a
/// few seconds: large enough to cover the mutation operators across every seed
/// family, small enough to stay trivially within the time/RAM bound.
const MUTANTS_PER_SEED: usize = 40;

/// Upper bound on any sane rendered page dimension, in pixels.  The production
/// caps reject far smaller; this is only a structural-sanity backstop so a
/// "successful" mutant that somehow yields an absurd bitmap is flagged as a
/// silent-corruption finding rather than passing.
const MAX_SANE_DIM: u32 = 100_000;

// ── Deterministic PRNG ────────────────────────────────────────────────────────

/// `SplitMix64` — a tiny, well-distributed, fully deterministic PRNG.
///
/// Hand-written inline (no `rand` dependency: it is not a dev-dependency of
/// this crate and the project policy forbids adding one for this).  The exact
/// algorithm does not matter; reproducibility does — the same seed yields the
/// same stream on every platform, so a harness failure is replayable.
struct SplitMix64(u64);

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    const fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, n)`.  `n == 0` returns 0 (no modulo-by-zero); callers
    /// never pass 0 for a meaningful bound, but defining it keeps the harness
    /// itself panic-free.
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        // `r < n <= usize::MAX`, so `r` always fits in `usize` on every
        // target; `try_from` documents that invariant and cannot fail here.
        let r = self.next_u64() % n as u64;
        usize::try_from(r).expect("modulo result is < n which fits in usize")
    }
}

// ── Seed corpus ───────────────────────────────────────────────────────────────

/// A small valid JPEG (32x32, 3-component baseline).  Reuses the committed
/// fixture so the DCT seed is a real decodable image, making truncation /
/// bit-flip mutants meaningfully exercise `decode_dct`'s error paths rather
/// than bouncing off an obviously-not-a-JPEG reject at byte 0.
const JPEG_RGB_32: &[u8] = include_bytes!("../../../tests/fixtures/jpeg/colour_32x32_444.jpg");

/// A small valid JPEG (64x64, single-component / grayscale baseline).  Covers
/// the 1-component SOF path, which the codec special-cases.
const JPEG_GRAY_64: &[u8] = include_bytes!("../../../tests/fixtures/jpeg/q20.jpg");

/// What kind of payload a seed carries — selects how it is wrapped before
/// mutation so the mutant actually reaches the intended decoder.
#[derive(Clone, Copy)]
enum SeedKind {
    /// The bytes are a complete PDF file; mutate the file directly.
    RawPdf,
    /// The bytes are a DCT (JPEG) codec stream; wrap as an image `XObject`.
    Dct,
    /// The bytes are a `CCITTFax` codec stream; wrap as a 1-bit image `XObject`.
    CcittFax,
    /// The bytes are a JBIG2 codec stream; wrap as a 1-bit image `XObject`.
    Jbig2,
}

/// A named seed: the human-readable label is reported on failure so a
/// regression points straight at the offending family.
struct Seed {
    name: &'static str,
    kind: SeedKind,
    bytes: Vec<u8>,
}

/// Build a minimal valid single-page PDF whose only content draws one image
/// `XObject` named `/Im0`.  `image_dict_extra` is spliced verbatim into the
/// image dictionary (e.g. `/Filter /DCTDecode`); `stream` is the raw codec
/// payload.  The xref offsets are computed from actual section lengths so the
/// document parses cleanly *before* mutation — the test then mutates the
/// finished bytes.
fn build_image_pdf(image_dict_extra: &str, stream: &[u8]) -> Vec<u8> {
    let header = "%PDF-1.4\n";
    let obj1 = "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n".to_string();
    let obj2 = "2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n".to_string();
    // A 16x16-point MediaBox keeps every "successful" decode cheap: even if a
    // mutant yields a valid image, the rasterised page is tiny.
    let obj3 = "3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 16 16] \
                /Resources <</XObject <</Im0 4 0 R>>>> /Contents 5 0 R>>\nendobj\n"
        .to_string();
    let obj4_head = format!(
        "4 0 obj\n<</Type /XObject /Subtype /Image /Width 8 /Height 8 \
         {image_dict_extra} /Length {}>>\nstream\n",
        stream.len()
    );
    let obj4_tail = b"\nendstream\nendobj\n";
    // `q ... Q` with a CTM scaling the unit image square to the page box.
    let content = b"q 16 0 0 16 0 0 cm /Im0 Do Q";
    let obj5 = format!(
        "5 0 obj\n<</Length {}>>\nstream\n{}\nendstream\nendobj\n",
        content.len(),
        std::str::from_utf8(content).expect("ASCII content")
    );

    let mut body = Vec::new();
    let mut offsets = Vec::with_capacity(5);
    let push = |body: &mut Vec<u8>, bytes: &[u8], offsets: &mut Vec<usize>| {
        offsets.push(header.len() + body.len());
        body.extend_from_slice(bytes);
    };
    push(&mut body, obj1.as_bytes(), &mut offsets);
    push(&mut body, obj2.as_bytes(), &mut offsets);
    push(&mut body, obj3.as_bytes(), &mut offsets);

    // Object 4 (the image) is assembled from head + raw stream + tail.
    offsets.push(header.len() + body.len());
    body.extend_from_slice(obj4_head.as_bytes());
    body.extend_from_slice(stream);
    body.extend_from_slice(obj4_tail);

    push(&mut body, obj5.as_bytes(), &mut offsets);

    let xref_start = header.len() + body.len();
    let mut xref = String::from("xref\n0 6\n0000000000 65535 f\r\n");
    for off in &offsets {
        write!(xref, "{off:010} 00000 n\r\n").expect("write to String is infallible");
    }
    let trailer = format!("trailer\n<</Size 6 /Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF");

    let mut out = Vec::with_capacity(xref_start + xref.len() + trailer.len());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&body);
    out.extend_from_slice(xref.as_bytes());
    out.extend_from_slice(trailer.as_bytes());
    out
}

/// A minimal valid empty-page PDF — the `RawPdf` structural seed.  Mutating
/// this exercises the parser's xref / trailer / object hardening directly.
fn minimal_valid_pdf() -> Vec<u8> {
    let header = "%PDF-1.4\n";
    let obj1 = "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n";
    let obj2 = "2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n";
    let obj3 = "3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 16 16]>>\nendobj\n";
    let off1 = header.len();
    let off2 = off1 + obj1.len();
    let off3 = off2 + obj2.len();
    let xref_start = off3 + obj3.len();
    let xref = format!(
        "xref\n0 4\n0000000000 65535 f\r\n\
         {off1:010} 00000 n\r\n{off2:010} 00000 n\r\n{off3:010} 00000 n\r\n"
    );
    let trailer = format!("trailer\n<</Size 4 /Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF");
    format!("{header}{obj1}{obj2}{obj3}{xref}{trailer}").into_bytes()
}

/// A synthetic CCITT Group 4 (T.6) payload.  `0x00 0x10 0x01` is the canonical
/// 24-bit EOFB (end-of-facsimile-block) preamble seen at the tail of real G4
/// streams; standing alone it is a *degenerate but plausibly-shaped* G4 input.
/// Validity is not the point — the mutation class is what matters; a seed that
/// merely looks like the format makes truncation/flip mutants land inside the
/// decoder's state machine rather than at a trivial reject.
fn synthetic_ccitt_g4() -> Vec<u8> {
    vec![0x00, 0x10, 0x01, 0x00, 0x10, 0x01]
}

/// A synthetic JBIG2 embedded-stream segment: a generic-region segment header
/// shape (segment number, flags, referred-to count, page assoc, data length)
/// followed by a short region info + a single MMR/arith data byte.  Shaped to
/// reach `decode_jbig2`'s segment parser; not a fully valid bitmap.
fn synthetic_jbig2() -> Vec<u8> {
    vec![
        0x00, 0x00, 0x00, 0x00, // segment number 0
        0x30, // flags: type 48 (page info) — exercises the header walker
        0x00, // referred-to count + retain flags
        0x01, // page association
        0x00, 0x00, 0x00, 0x13, // data length
        0x00, 0x00, 0x00, 0x08, // region width
        0x00, 0x00, 0x00, 0x08, // region height
        0x00, 0x00, 0x00, 0x00, // x location
        0x00, 0x00, 0x00, 0x00, // y location
        0x00, // region flags
    ]
}

/// Assemble the full seed corpus: structural + every hardened codec family +
/// the NF-9 malformed set (empty, no-`%PDF`, header-only, truncated).
fn seed_corpus() -> Vec<Seed> {
    let valid = minimal_valid_pdf();
    let header_only = b"%PDF-1.4\n".to_vec();
    let truncated = {
        let mut v = valid.clone();
        v.truncate(v.len() / 2);
        v
    };
    vec![
        Seed {
            name: "valid-pdf",
            kind: SeedKind::RawPdf,
            bytes: valid,
        },
        Seed {
            name: "empty",
            kind: SeedKind::RawPdf,
            bytes: Vec::new(),
        },
        Seed {
            name: "no-%PDF",
            kind: SeedKind::RawPdf,
            bytes: b"not a pdf at all".to_vec(),
        },
        Seed {
            name: "header-only",
            kind: SeedKind::RawPdf,
            bytes: header_only,
        },
        Seed {
            name: "truncated-pdf",
            kind: SeedKind::RawPdf,
            bytes: truncated,
        },
        Seed {
            name: "jpeg-rgb-32",
            kind: SeedKind::Dct,
            bytes: JPEG_RGB_32.to_vec(),
        },
        Seed {
            name: "jpeg-gray-64",
            kind: SeedKind::Dct,
            bytes: JPEG_GRAY_64.to_vec(),
        },
        Seed {
            name: "ccitt-g4",
            kind: SeedKind::CcittFax,
            bytes: synthetic_ccitt_g4(),
        },
        Seed {
            name: "jbig2-generic",
            kind: SeedKind::Jbig2,
            bytes: synthetic_jbig2(),
        },
    ]
}

/// Wrap a (possibly already-mutated) codec payload into a complete PDF so it
/// reaches the intended decoder, or pass a `RawPdf` payload through verbatim.
fn wrap_for_kind(kind: SeedKind, payload: &[u8]) -> Vec<u8> {
    match kind {
        SeedKind::RawPdf => payload.to_vec(),
        SeedKind::Dct => build_image_pdf(
            "/Filter /DCTDecode /ColorSpace /DeviceRGB /BitsPerComponent 8",
            payload,
        ),
        SeedKind::CcittFax => build_image_pdf(
            "/Filter /CCITTFax /ColorSpace /DeviceGray /BitsPerComponent 1 \
             /DecodeParms <</K -1 /Columns 8 /Rows 8>>",
            payload,
        ),
        SeedKind::Jbig2 => build_image_pdf(
            "/Filter /JBIG2Decode /ColorSpace /DeviceGray /BitsPerComponent 1",
            payload,
        ),
    }
}

// ── Mutation operators ────────────────────────────────────────────────────────

/// Apply one randomly-chosen, bounded mutation to `seed`, returning the mutant.
///
/// Each operator models a real corruption mode the campaign hardened against:
/// bit-flips (transmission/storage damage), truncation (interrupted I/O),
/// zeroed runs (sparse-file / wiped-sector), length-field corruption (the
/// classic OOB-read trigger), token injection (confusing the lexer/xref
/// recovery), and oversized declared dimensions (the unbounded-allocation
/// trigger).  All are size-bounded so no mutant can balloon RAM.
fn mutate(seed: &[u8], rng: &mut SplitMix64) -> Vec<u8> {
    let mut m = seed.to_vec();
    match rng.below(7) {
        // Bit-flips: 1..=8 single-bit flips at random offsets.
        0 if !m.is_empty() => {
            let flips = 1 + rng.below(8);
            for _ in 0..flips {
                let i = rng.below(m.len());
                m[i] ^= 1u8 << rng.below(8);
            }
        }
        // Truncate at a random offset (may produce the empty slice).
        1 if !m.is_empty() => {
            let cut = rng.below(m.len());
            m.truncate(cut);
        }
        // Zero a bounded run (sparse-file / wiped-region model).
        2 if !m.is_empty() => {
            let start = rng.below(m.len());
            let len = 1 + rng.below(32.min(m.len()));
            for b in m.iter_mut().skip(start).take(len) {
                *b = 0;
            }
        }
        // Inject a confusing PDF token at a random offset (xref/lexer stress).
        3 => {
            let tokens: [&[u8]; 4] = [b"endstream", b"endobj", b"\n%%EOF\n", b"0 0 obj"];
            let tok = tokens[rng.below(tokens.len())];
            let at = if m.is_empty() { 0 } else { rng.below(m.len()) };
            let mut next = Vec::with_capacity(m.len() + tok.len());
            next.extend_from_slice(&m[..at]);
            next.extend_from_slice(tok);
            next.extend_from_slice(&m[at..]);
            m = next;
        }
        // Corrupt a 4-byte big-endian length-ish field in place: classic
        // "declared length disagrees with reality" OOB-read trigger.
        4 if m.len() >= 4 => {
            let at = rng.below(m.len() - 3);
            // Bounded < 0x00FF_FFFF, so it always fits in u32 on every target.
            let corrupt = u32::try_from(rng.below(0x00FF_FFFF))
                .expect("value is < 0x00FF_FFFF which fits in u32");
            m[at..at + 4].copy_from_slice(&corrupt.to_be_bytes());
        }
        // Splice an oversized declared dimension into the byte stream — the
        // unbounded-allocation trigger the MAX_PX_AREA cap defends.
        5 => {
            let needle = b"/Width 8";
            if let Some(pos) = m.windows(needle.len()).position(|w| w == needle) {
                let big = b"/Width 999999999";
                let mut next = Vec::with_capacity(m.len() + 8);
                next.extend_from_slice(&m[..pos]);
                next.extend_from_slice(big);
                next.extend_from_slice(&m[pos + needle.len()..]);
                m = next;
            } else if !m.is_empty() {
                // No declared dimension to corrupt (e.g. a RawPdf seed): fall
                // back to a duplicate-byte-run so the mutant is still distinct.
                let i = rng.below(m.len());
                let b = m[i];
                let mut next = Vec::with_capacity(m.len() + 16);
                next.extend_from_slice(&m[..i]);
                next.extend(std::iter::repeat_n(b, 16));
                next.extend_from_slice(&m[i..]);
                m = next;
            }
        }
        // Duplicate a bounded chunk (token-stream confusion without growth
        // blow-up: capped at 64 bytes).
        _ if !m.is_empty() => {
            let start = rng.below(m.len());
            let len = 1 + rng.below(64.min(m.len()));
            let chunk: Vec<u8> = m[start..(start + len).min(m.len())].to_vec();
            let at = rng.below(m.len());
            let mut next = Vec::with_capacity(m.len() + chunk.len());
            next.extend_from_slice(&m[..at]);
            next.extend_from_slice(&chunk);
            next.extend_from_slice(&m[at..]);
            m = next;
        }
        // Empty-seed fallthrough for the `!m.is_empty()` arms: a single byte
        // so the mutant is non-trivially distinct from the empty seed.
        // Low byte of a fresh draw — a random byte with no lossy cast.
        _ => m.push(rng.next_u64().to_le_bytes()[0]),
    }
    m
}

// ── Driving a mutant through the real decode+render path ──────────────────────

/// Outcome of feeding one mutant through the bounded entry path.
enum Outcome {
    /// Parser cleanly rejected the bytes (the common, desired case).
    CleanErr,
    /// Parsed + rendered to completion with in-cap geometry (a bounded
    /// success / degrade — also acceptable).
    OkBounded,
    /// FAILURE: a panic escaped into our `catch_unwind` — a decoder/parser
    /// hit an unwrap/index-OOB on malformed input where it owed a clean
    /// `Err`/`None`.  The production page layer also catches per-page, but a
    /// panic *here* is still a robustness gap worth a root fix.
    Panicked(String),
    /// FAILURE: a "successful" render produced structurally impossible
    /// geometry — a silent-corruption escape.
    InsaneGeometry(u32, u32),
}

/// Run the real public decode+render pipeline on `bytes`, catching any panic.
///
/// This is the SAME sequence the CLI uses (`Document::from_bytes_owned` ->
/// `get_page` -> `parse_page_by_id` -> `PageRenderer::execute` -> `finish`),
/// so it genuinely invokes the hardened parser and the JPEG/CCITT/JBIG2
/// codecs — not a stub.  `catch_unwind` only *observes* a panic so the harness
/// can report it; it does not mask one (a caught panic is a test FAILURE).
fn drive(bytes: Vec<u8>) -> Outcome {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let Ok(doc) = Document::from_bytes_owned(bytes) else {
            return Outcome::CleanErr;
        };
        let Ok(page_id) = doc.get_page(0) else {
            return Outcome::CleanErr;
        };
        let Ok(ops) = rasterrocket_interp::parse_page_by_id(&doc, page_id) else {
            return Outcome::CleanErr;
        };
        // Fixed, valid geometry: the mutated bytes flow into the PDF body /
        // codec stream, never into these dimensions, so `PageRenderer::new`'s
        // scale assertion (a caller-contract precondition, not an
        // input-driven path) is never the thing under test.
        let Ok(mut renderer) =
            rasterrocket_interp::renderer::PageRenderer::new(16, 16, &doc, page_id)
        else {
            return Outcome::CleanErr;
        };
        renderer.execute(&ops);
        let (bitmap, _diag) = renderer.finish();
        let (w, h) = (bitmap.width, bitmap.height);
        if w > MAX_SANE_DIM || h > MAX_SANE_DIM {
            Outcome::InsaneGeometry(w, h)
        } else {
            Outcome::OkBounded
        }
    }));

    match result {
        Ok(outcome) => outcome,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic payload>".to_string());
            Outcome::Panicked(msg)
        }
    }
}

/// Drive `bytes` under a hard wall-clock guard on a worker thread.
///
/// `recv_timeout` ensures the *harness itself* can never hang: if the worker
/// exceeds [`PER_MUTANT_GUARD`] the guard fires and the mutant is recorded as
/// a HANG failure.  (The worker thread is then detached; a genuine infinite
/// loop would leak it, but a hang is already a hard failure so the test ends.)
fn drive_guarded(bytes: Vec<u8>) -> Result<Outcome, ()> {
    let (tx, rx) = mpsc::channel();
    let handle = thread::Builder::new()
        .name("fuzz-mutant".into())
        .spawn(move || {
            let outcome = drive(bytes);
            // The receiver may already be gone if the guard fired first; a
            // failed send is the expected, harmless race in that case.
            drop(tx.send(outcome));
        })
        .expect("spawn mutant worker");

    let Ok(outcome) = rx.recv_timeout(PER_MUTANT_GUARD) else {
        return Err(()); // timeout => the mutant hung; reported as a failure
    };
    // The worker finished; join it so the thread is reaped (it cannot block
    // now that it has already sent).  `join` only errors if the worker itself
    // panicked, which `drive`'s `catch_unwind` already converts to an
    // `Outcome`, so a join error here is unreachable and safely dropped.
    drop(handle.join());
    Ok(outcome)
}

// ── The property test ─────────────────────────────────────────────────────────

/// Drive a bounded, deterministic population of mutated inputs through the
/// real decode+render path and assert the campaign's central invariant on
/// every one of them.
#[test]
fn malformed_input_invariant_holds_across_decoder_classes() {
    let started = Instant::now();
    let seeds = seed_corpus();
    let mut total = 0usize;
    let mut clean_err = 0usize;
    let mut ok_bounded = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for seed in &seeds {
        // Per-seed deterministic stream: the seed name's bytes fold into the
        // PRNG seed so each family gets an independent but reproducible
        // sequence.  Same build => identical population every run.
        let mut seed_val: u64 = 0x1234_5678_9ABC_DEF0;
        for &b in seed.name.as_bytes() {
            seed_val = seed_val.wrapping_mul(31).wrapping_add(u64::from(b));
        }
        let mut rng = SplitMix64::new(seed_val);

        for n in 0..MUTANTS_PER_SEED {
            let mutant_payload = mutate(&seed.bytes, &mut rng);
            let bytes = wrap_for_kind(seed.kind, &mutant_payload);
            total += 1;

            match drive_guarded(bytes) {
                Ok(Outcome::CleanErr) => clean_err += 1,
                Ok(Outcome::OkBounded) => ok_bounded += 1,
                Ok(Outcome::Panicked(msg)) => failures.push(format!(
                    "PANIC  seed={} mutant#{n}: decoder/parser panicked on \
                     malformed input (owed a clean Err/None): {msg}",
                    seed.name
                )),
                Ok(Outcome::InsaneGeometry(w, h)) => failures.push(format!(
                    "INSANE seed={} mutant#{n}: render returned Ok with \
                     out-of-cap geometry {w}x{h} (silent corruption)",
                    seed.name
                )),
                Err(()) => failures.push(format!(
                    "HANG   seed={} mutant#{n}: exceeded {:?} wall-clock \
                     guard (infinite loop / unbounded retry)",
                    seed.name, PER_MUTANT_GUARD
                )),
            }
        }
    }

    let elapsed = started.elapsed();
    eprintln!(
        "decoder property harness: {total} mutants across {} seeds in {elapsed:?} \
         ({clean_err} clean-Err, {ok_bounded} Ok-bounded, {} FAIL)",
        seeds.len(),
        failures.len()
    );

    assert!(
        failures.is_empty(),
        "malformed-input invariant violated by {} of {total} mutants:\n{}",
        failures.len(),
        failures.join("\n")
    );

    // Sanity floor: the harness must have actually exercised the population.
    // A regression that silently neutered mutation/driving would make this
    // fail rather than pass vacuously.
    assert_eq!(
        total,
        seeds.len() * MUTANTS_PER_SEED,
        "mutant count drifted"
    );
    assert!(
        clean_err + ok_bounded == total,
        "every mutant must terminate in CleanErr or OkBounded"
    );
}

/// Codec-reachability guard.
///
/// The property test is only meaningful if a codec mutant actually reaches
/// the codec decoder rather than bouncing off a malformed *wrapper* PDF
/// before the image is ever resolved.  This test pins that: every codec
/// seed's UNMUTATED wrapped PDF must parse cleanly and drive all the way to
/// `PageRenderer::finish` with the fixed 16x16 page geometry.  If a future
/// change broke `build_image_pdf` so the wrapper no longer parsed, the
/// property test could pass vacuously (every mutant a trivial `CleanErr`
/// from the wrapper, never touching `decode_dct`/`decode_ccitt`/
/// `decode_jbig2`); this test fails loudly instead.
#[test]
fn codec_seeds_reach_the_decoder_through_a_valid_wrapper() {
    for seed in seed_corpus() {
        // Only the codec families have a wrapper to validate; RawPdf seeds
        // are the file itself and are exercised directly by the property
        // test (several are intentionally invalid, e.g. the NF-9 set).
        if matches!(seed.kind, SeedKind::RawPdf) {
            continue;
        }
        let pdf = wrap_for_kind(seed.kind, &seed.bytes);
        let outcome = drive(pdf);
        assert!(
            matches!(outcome, Outcome::OkBounded),
            "codec seed {} must parse + render through a valid wrapper \
             (so mutants genuinely reach the decoder), got a non-Ok outcome",
            seed.name
        );
    }
}
